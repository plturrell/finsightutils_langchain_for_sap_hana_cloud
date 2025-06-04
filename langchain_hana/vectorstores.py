"""SAP HANA Cloud Vector Engine"""

from __future__ import annotations

import json
import logging
import re
import struct
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Pattern,
    Type,
)

import numpy as np
from hdbcli import dbapi  # type: ignore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance

from langchain_hana.embeddings import HanaInternalEmbeddings
from langchain_hana.query_constructors import (
    CONTAINS_OPERATOR,
    LOGICAL_OPERATORS_TO_SQL,
    CreateWhereClause,
)
from langchain_hana.utils import DistanceStrategy

logger = logging.getLogger(__name__)

HANA_DISTANCE_FUNCTION: dict = {
    DistanceStrategy.COSINE: ("COSINE_SIMILARITY", "DESC"),
    DistanceStrategy.EUCLIDEAN_DISTANCE: ("L2DISTANCE", "ASC"),
}

VECTOR_COLUMN_SQL_TYPES = ["REAL_VECTOR", "HALF_VECTOR"]

INTERMEDIATE_TABLE_NAME = "intermediate_result"

default_distance_strategy = DistanceStrategy.COSINE
default_table_name: str = "EMBEDDINGS"
default_content_column: str = "VEC_TEXT"
default_metadata_column: str = "VEC_META"
default_vector_column: str = "VEC_VECTOR"
default_vector_column_length: int = -1  # -1 means dynamic length
default_vector_column_type: str = "REAL_VECTOR"


class HanaDB(VectorStore):
    """SAP HANA Cloud Vector Engine

    The prerequisite for using this class is the installation of the ``hdbcli``
    Python package.

    The HanaDB vectorstore can be created by providing an embedding function and
    an existing database connection. Optionally, the names of the table and the
    columns to use.
    """

    def __init__(
        self,
        connection: dbapi.Connection,
        embedding: Embeddings,
        distance_strategy: DistanceStrategy = default_distance_strategy,
        table_name: str = default_table_name,
        content_column: str = default_content_column,
        metadata_column: str = default_metadata_column,
        vector_column: str = default_vector_column,
        vector_column_length: int = default_vector_column_length,
        vector_column_type: str = default_vector_column_type,
        *,
        specific_metadata_columns: Optional[list[str]] = None,
    ):
        valid_distance = False
        for key in HANA_DISTANCE_FUNCTION.keys():
            if key is distance_strategy:
                valid_distance = True
        if not valid_distance:
            available_strategies = ", ".join([str(s) for s in HANA_DISTANCE_FUNCTION.keys()])
            raise ValueError(
                f"Unsupported distance_strategy: {distance_strategy}. "
                f"Please use one of the following strategies from the DistanceStrategy enum: {available_strategies}. "
                f"For most use cases, DistanceStrategy.COSINE is recommended."
            )

        self.connection = connection
        self.distance_strategy = distance_strategy
        self.table_name = HanaDB._sanitize_name(table_name)
        self.content_column = HanaDB._sanitize_name(content_column)
        self.metadata_column = HanaDB._sanitize_name(metadata_column)
        self.vector_column = HanaDB._sanitize_name(vector_column)
        self.vector_column_length = HanaDB._sanitize_int(vector_column_length)
        self.vector_column_type = HanaDB._sanitize_vector_column_type(
            vector_column_type, connection
        )
        self.specific_metadata_columns = HanaDB._sanitize_specific_metadata_columns(
            specific_metadata_columns or []
        )

        # Configure the embedding (internal or external)
        # - Internal embeddings: Use SAP HANA's built-in VECTOR_EMBEDDING function (faster, more efficient)
        # - External embeddings: Use a Python-based embedding model (more flexible, works with any model)
        self.embedding: Embeddings
        self.use_internal_embeddings: bool = False  # Flag indicating if we're using internal embeddings
        self.internal_embedding_model_id: str = ""   # Model ID for internal embeddings
        self.set_embedding(embedding)  # Configure the embedding approach based on the provided instance

        # Initialize the table if it doesn't exist
        self._initialize_table()

    def set_embedding(self, embedding: Embeddings) -> None:
        """
        Configure the embedding approach for this vector store instance.
        
        This method determines whether to use internal (SAP HANA native) or external 
        (Python-based) embeddings based on the type of the provided embedding instance.
        
        Args:
            embedding: An instance of a LangChain Embeddings class. If this is an instance
                      of HanaInternalEmbeddings, the vector store will use HANA's internal
                      VECTOR_EMBEDDING function for all embedding operations. Otherwise,
                      it will use the provided embedding model for generating embeddings
                      in Python.
                      
        Notes:
            - Internal embeddings (HanaInternalEmbeddings) are generally faster and more 
              efficient as they leverage SAP HANA's native capabilities
            - External embeddings offer more flexibility in model selection
            - When using internal embeddings, this method validates that the specified
              embedding function exists in your HANA instance
            
        Examples:
            ```python
            # Using internal embeddings
            from langchain_hana.embeddings import HanaInternalEmbeddings
            store = HanaDB(
                connection=conn,
                embedding=HanaInternalEmbeddings(internal_embedding_model_id="SAP_NEB.20240715")
            )
            
            # Using external embeddings
            from langchain_core.embeddings import HuggingFaceEmbeddings
            store = HanaDB(
                connection=conn,
                embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            )
            
            # Changing embeddings after initialization
            store.set_embedding(new_embedding_model)
            ```
        """
        self.embedding = embedding
        # Decide whether to use internal or external embeddings
        if isinstance(embedding, HanaInternalEmbeddings):
            # Internal embeddings - use SAP HANA's VECTOR_EMBEDDING function
            self.use_internal_embeddings = True
            self.internal_embedding_model_id = embedding.get_model_id()
            self._validate_internal_embedding_function()  # Ensure the embedding function exists
        else:
            # External embeddings - use Python-based embedding model
            self.use_internal_embeddings = False
            self.internal_embedding_model_id = ""

    def _initialize_table(self) -> None:
        """Create the table if it doesn't exist and validate columns."""
        if not self._table_exists(self.table_name):
            # Using parametrized query with named parameters for table creation
            params = {
                "content_column": self.content_column,
                "metadata_column": self.metadata_column,
                "vector_column": self.vector_column,
                "vector_column_type": self.vector_column_type,
            }
            
            sql_str = (
                'CREATE TABLE "?" '
                '("?" NCLOB, '
                '"?" NCLOB, '
                '"?" ? '
            )
            
            if self.vector_column_length in [-1, 0]:
                sql_str += ");"
            else:
                sql_str += "(?))"
                params["vector_column_length"] = self.vector_column_length

            try:
                cur = self.connection.cursor()
                # Using standard HANA parametrized query execution
                cur.execute(sql_str, 
                           self.table_name,
                           params["content_column"],
                           params["metadata_column"],
                           params["vector_column"],
                           params["vector_column_type"],
                           params.get("vector_column_length") if self.vector_column_length not in [-1, 0] else None
                           )
            finally:
                cur.close()

        # Check if the needed columns exist and have the correct type
        self._check_column(self.table_name, self.content_column, ["NCLOB", "NVARCHAR"])
        self._check_column(self.table_name, self.metadata_column, ["NCLOB", "NVARCHAR"])
        self._check_column(
            self.table_name,
            self.vector_column,
            [self.vector_column_type],
            self.vector_column_length,
        )
        for column_name in self.specific_metadata_columns:
            self._check_column(self.table_name, column_name)

    def _table_exists(self, table_name) -> bool:  # type: ignore[no-untyped-def]
        sql_str = (
            "SELECT COUNT(*) FROM SYS.TABLES WHERE SCHEMA_NAME = CURRENT_SCHEMA"
            " AND TABLE_NAME = ?"
        )
        try:
            cur = self.connection.cursor()
            cur.execute(sql_str, (table_name))
            if cur.has_result_set():
                rows = cur.fetchall()
                if rows[0][0] == 1:
                    return True
        finally:
            cur.close()
        return False

    def _check_column(  # type: ignore[no-untyped-def]
        self, table_name, column_name, column_type=None, column_length=None
    ):
        sql_str = (
            "SELECT DATA_TYPE_NAME, LENGTH FROM SYS.TABLE_COLUMNS WHERE "
            "SCHEMA_NAME = CURRENT_SCHEMA "
            "AND TABLE_NAME = ? AND COLUMN_NAME = ?"
        )
        try:
            cur = self.connection.cursor()
            cur.execute(sql_str, (table_name, column_name))
            if cur.has_result_set():
                rows = cur.fetchall()
                if len(rows) == 0:
                    raise AttributeError(f"Column {column_name} does not exist")
                # Check data type
                if column_type:
                    if rows[0][0] not in column_type:
                        raise AttributeError(
                            f"Column {column_name} has the wrong type: {rows[0][0]}"
                        )
                # Check length, if parameter was provided
                # Length can either be -1 (QRC01+02-24) or 0 (QRC03-24 onwards)
                # to indicate no length constraint being present.
                if column_length is not None and column_length > 0:
                    if rows[0][1] != column_length:
                        raise AttributeError(
                            f"Column {column_name} has the wrong length: {rows[0][1]} "
                            f"expected: {column_length}"
                        )
            else:
                raise AttributeError(f"Column {column_name} does not exist")
        finally:
            cur.close()

    def _validate_internal_embedding_function(self) -> None:
        """
        Validate that the specified internal embedding function exists and works in the database.
        
        This method tests the VECTOR_EMBEDDING function with the provided model ID by
        executing a simple query against the database. This helps catch configuration
        issues early, before attempting to use the embedding function in actual operations.
        
        Potential issues that this validation can detect:
        1. The VECTOR_EMBEDDING function doesn't exist (older SAP HANA versions)
        2. The specified model ID is invalid or not available
        3. Insufficient permissions to execute the VECTOR_EMBEDDING function
        4. Database connectivity issues
        
        Raises:
            ValueError: If the embedding model ID is None
            Various database exceptions: If the VECTOR_EMBEDDING function doesn't exist,
                                        the model ID is invalid, or there are permission issues
        
        Note:
            This method is automatically called by set_embedding when an instance
            of HanaInternalEmbeddings is provided.
        """
        # Validate that a model ID was provided
        if self.internal_embedding_model_id is None:
            raise ValueError(
                "Internal embedding model ID cannot be None. "
                "When using HanaInternalEmbeddings, you must specify a valid model ID such as 'SAP_NEB.20240715'. "
                "Check SAP HANA Cloud documentation for available embedding models in your environment."
            )
        
        # Execute a test query to verify the VECTOR_EMBEDDING function works with the provided model ID
        cur = self.connection.cursor()
        try:
            # We wrap the result in TO_NVARCHAR and COUNT to ensure we get a result
            # even if the function returns NULL or multiple rows
            cur.execute(
                (
                    "SELECT COUNT(TO_NVARCHAR("
                    "VECTOR_EMBEDDING('test', 'QUERY', :model_version))) "
                    ' AS "CNT" FROM sys.DUMMY;'
                ),
                model_version=self.internal_embedding_model_id,
            )
            # Note: We don't need to check the result - if the function doesn't exist
            # or the model ID is invalid, an exception will be raised
        finally:
            cur.close()

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    @staticmethod
    def _sanitize_name(input_str: str) -> str:  # type: ignore[misc]
        # Remove characters that are not alphanumeric or underscores
        return re.sub(r"[^a-zA-Z0-9_]", "", input_str)

    @staticmethod
    def _sanitize_int(input_int: any) -> int:  # type: ignore[valid-type]
        value = int(str(input_int))
        if value < -1:
            raise ValueError(f"Value ({value}) must not be smaller than -1")
        return int(str(input_int))

    @staticmethod
    def _sanitize_list_float(embedding: list[float]) -> list[float]:
        for value in embedding:
            if not isinstance(value, float):
                raise ValueError(f"Value ({value}) does not have type float")
        return embedding

    # Compile pattern only once, for better performance
    _compiled_pattern: Pattern = re.compile("^[_a-zA-Z][_a-zA-Z0-9]*$")

    @staticmethod
    def _sanitize_metadata_keys(metadata: dict) -> dict:
        for key in metadata.keys():
            if not HanaDB._compiled_pattern.match(key):
                raise ValueError(f"Invalid metadata key {key}")

        return metadata

    @staticmethod
    def _sanitize_specific_metadata_columns(
        specific_metadata_columns: list[str],
    ) -> list[str]:
        metadata_columns = []
        for c in specific_metadata_columns:
            sanitized_name = HanaDB._sanitize_name(c)
            metadata_columns.append(sanitized_name)
        return metadata_columns

    @staticmethod
    def _sanitize_vector_column_type(
        vector_column_type: str, connection: dbapi.Connection
    ) -> str:
        vector_column_type_upper = vector_column_type.upper()
        if vector_column_type_upper not in VECTOR_COLUMN_SQL_TYPES:
            raise ValueError(
                f"Unsupported vector_column_type: {vector_column_type}. "
                f"Must be one of {', '.join(VECTOR_COLUMN_SQL_TYPES)}.\n"
                f"- REAL_VECTOR: Standard 32-bit float vector type (available in HANA Cloud QRC 1/2024+)\n"
                f"- HALF_VECTOR: Compressed 16-bit float vector type, uses less storage (available in HANA Cloud QRC 2/2025+)\n"
                f"For most use cases, REAL_VECTOR is recommended for compatibility and precision."
            )
        HanaDB._validate_datatype_support(connection, vector_column_type_upper)
        return vector_column_type_upper

    @staticmethod
    def _get_min_supported_version(datatype: str) -> str:
        if datatype == "HALF_VECTOR":
            return "2025.15 (QRC 2/2025)"
        elif datatype == "REAL_VECTOR":
            return "2024.2 (QRC 1/2024)"
        else:
            raise ValueError(f"Unknown datatype: '{datatype}'")

    @staticmethod
    def _get_instance_version(connection: dbapi.Connection) -> Optional[str]:
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT CLOUD_VERSION FROM SYS.M_DATABASE;")
            result = cursor.fetchone()
            return result[0]
        except dbapi.Error:
            return None
        finally:
            cursor.close()

    @staticmethod
    def _get_available_datatypes(connection: dbapi.Connection) -> set:
        cur = connection.cursor()
        try:
            cur.execute("SELECT TYPE_NAME FROM SYS.DATA_TYPES")
            if cur.has_result_set():
                rows = cur.fetchall()
                available_types = {row[0] for row in rows}
                return available_types
            raise ValueError("No data types returned by the database.")
        finally:
            cur.close()

    @staticmethod
    def _validate_datatype_support(connection: dbapi.Connection, datatype: str) -> bool:
        if datatype in HanaDB._get_available_datatypes(connection):
            return True

        # Get instance version, but don't include it in error if retrieval fails
        instance_version = HanaDB._get_instance_version(connection)
        min_instance_version = HanaDB._get_min_supported_version(datatype)
        
        error_message = (
            f"Vector type '{datatype}' is not available on your SAP HANA Cloud instance.\n\n"
            f"Issue: Your database does not support the {datatype} data type.\n"
        )
        
        if instance_version:
            error_message += f"Your current instance version: {instance_version}\n"
        
        error_message += (
            f"Required version: {min_instance_version}\n\n"
            f"Solutions:\n"
            f"1. Use 'REAL_VECTOR' instead (supported in older versions)\n"
            f"2. Upgrade your SAP HANA Cloud instance to {min_instance_version} or newer\n"
            f"3. Contact your SAP HANA Cloud administrator to check for available vector types"
        )

        raise ValueError(error_message)

    def _serialize_binary_format(self, values: list[float]) -> bytes:
        # Converts a list of floats into binary format
        if self.vector_column_type == "HALF_VECTOR":
            # 2-byte half-precision float serialization
            return struct.pack(f"<I{len(values)}e", len(values), *values)
        elif self.vector_column_type == "REAL_VECTOR":
            # 4-byte float serialization (standard FVECS format)
            return struct.pack(f"<I{len(values)}f", len(values), *values)
        else:
            raise ValueError(
                f"Unsupported vector column type: {self.vector_column_type}"
            )

    def _deserialize_binary_format(self, fvecs: bytes) -> list[float]:
        # Extracts a list of floats from binary format
        dim = struct.unpack_from("<I", fvecs, 0)[0]
        if self.vector_column_type == "HALF_VECTOR":
            # 2-byte half-precision float deserialization
            return list(struct.unpack_from(f"<{dim}e", fvecs, 4))
        elif self.vector_column_type == "REAL_VECTOR":
            # 4-byte float deserialization (standard FVECS format)
            return list(struct.unpack_from(f"<{dim}f", fvecs, 4))
        else:
            raise ValueError(
                f"Unsupported vector column type: {self.vector_column_type}"
            )

    def _split_off_special_metadata(self, metadata: dict) -> tuple[dict, list]:
        # Use provided values by default or fallback
        special_metadata = []

        if not metadata:
            return {}, []

        for column_name in self.specific_metadata_columns:
            special_metadata.append(metadata.get(column_name, None))

        return metadata, special_metadata

    def _convert_to_target_vector_type(self, expr: str) -> str:
        """
        Converts a vector expression to the target vector column type.

        Applies the appropriate vector conversion function
            (TO_REAL_VECTOR or TO_HALF_VECTOR) to the provided
            expression based on the configured vector_column_type.

        Args:
            expr (str): a vector expression

        Returns:
            str: The expression wrapped with the appropriate conversion function.
        """
        if self.vector_column_type in VECTOR_COLUMN_SQL_TYPES:
            return f"TO_{self.vector_column_type}('{expr}')"
        else:
            raise ValueError(f"Unsupported vector type: {self.vector_column_type}")

    def _convert_vector_embedding_to_column_type(self, expr: str) -> str:
        """
        Makes sure that an embedding produced by HANA's VECTOR_EMBEDDING
        aligns with the column type.

        Note that VECTOR_EMBEDDING always returns REAL_VECTORs.

        Args:
            expr (str): SQL expression producing an embedding vector.
        Returns:
            str: Wrapped expression if vector column type is not REAL_VECTOR,
                otherwise the original expression.
        """

        if "VECTOR_EMBEDDING" not in expr.upper():
            raise ValueError(f"Expected 'VECTOR_EMBEDDING' in '{expr}'")

        if self.vector_column_type != "REAL_VECTOR":
            return self._convert_to_target_vector_type(expr)

        return expr

    def create_hnsw_index(
        self,
        m: Optional[int] = None,  # Optional M parameter
        ef_construction: Optional[int] = None,  # Optional efConstruction parameter
        ef_search: Optional[int] = None,  # Optional efSearch parameter
        index_name: Optional[str] = None,  # Optional custom index name
    ) -> None:
        """
        Creates an HNSW vector index on a specified table and vector column with
        optional build and search configurations. If no configurations are provided,
        default parameters from the database are used. If provided values exceed the
        valid ranges, an error will be raised.
        The index is always created in ONLINE mode.

        Args:
            m: (Optional) Maximum number of neighbors per graph node
                (Valid Range: [4, 1000])
            ef_construction: (Optional) Maximal candidates to consider when building
                                the graph (Valid Range: [1, 100000])
            ef_search: (Optional) Minimum candidates for top-k-nearest neighbor
                                queries (Valid Range: [1, 100000])
            index_name: (Optional) Custom index name. Defaults to
                        <table_name>_<distance_strategy>_idx
        """
        # Set default index name if not provided
        distance_func_name = HANA_DISTANCE_FUNCTION[self.distance_strategy][0]
        default_index_name = f"{self.table_name}_{distance_func_name}_idx"
        # Use provided index_name or default
        index_name = (
            HanaDB._sanitize_name(index_name) if index_name else default_index_name
        )
        # Initialize build_config and search_config as empty dictionaries
        build_config = {}
        search_config = {}

        # Validate and add m parameter to build_config if provided
        if m is not None:
            m = HanaDB._sanitize_int(m)
            if not (4 <= m <= 1000):
                raise ValueError("M must be in the range [4, 1000]")
            build_config["M"] = m

        # Validate and add ef_construction to build_config if provided
        if ef_construction is not None:
            ef_construction = HanaDB._sanitize_int(ef_construction)
            if not (1 <= ef_construction <= 100000):
                raise ValueError("efConstruction must be in the range [1, 100000]")
            build_config["efConstruction"] = ef_construction

        # Validate and add ef_search to search_config if provided
        if ef_search is not None:
            ef_search = HanaDB._sanitize_int(ef_search)
            if not (1 <= ef_search <= 100000):
                raise ValueError("efSearch must be in the range [1, 100000]")
            search_config["efSearch"] = ef_search

        # Convert build_config and search_config to JSON strings if they contain values
        build_config_str = json.dumps(build_config) if build_config else ""
        search_config_str = json.dumps(search_config) if search_config else ""

        # Create the index SQL string with the ONLINE keyword
        sql_str = (
            f'CREATE HNSW VECTOR INDEX {index_name} ON "{self.table_name}" '
            f'("{self.vector_column}") '
            f"SIMILARITY FUNCTION {distance_func_name} "
        )

        # Append build_config to the SQL string if provided
        if build_config_str:
            sql_str += f"BUILD CONFIGURATION '{build_config_str}' "

        # Append search_config to the SQL string if provided
        if search_config_str:
            sql_str += f"SEARCH CONFIGURATION '{search_config_str}' "

        # Always add the ONLINE option
        sql_str += "ONLINE "
        cur = self.connection.cursor()
        try:
            cur.execute(sql_str)
        finally:
            cur.close()

    def add_texts(  # type: ignore[override]
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts to the vectorstore.

        Args:
            texts (Iterable[str]): Iterable of strings/text to add to the vectorstore.
            metadatas (Optional[list[dict]], optional): Optional list of metadatas.
                Defaults to None.
            embeddings (Optional[list[list[float]]], optional): Optional pre-generated
                embeddings. Defaults to None.

        Returns:
            list[str]: empty list
        """

        # decide how to add texts
        # using external embedding instance or internal embedding function of HanaDB
        if self.use_internal_embeddings:
            return self._add_texts_using_internal_embedding(
                texts, metadatas, embeddings
            )
        else:
            return self._add_texts_using_external_embedding(
                texts, metadatas, embeddings
            )

    def _add_texts_using_external_embedding(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts using external embedding function"""
        # Create all embeddings of the texts beforehand to improve performance
        if embeddings is None:
            embeddings = self.embedding.embed_documents(list(texts))

        # Create sql parameters array
        sql_params = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            metadata, extracted_special_metadata = self._split_off_special_metadata(
                metadata
            )
            sql_params.append(
                (
                    text,
                    json.dumps(HanaDB._sanitize_metadata_keys(metadata)),
                    self._serialize_binary_format(embeddings[i]),
                    *extracted_special_metadata,
                )
            )

        specific_metadata_columns_string = self._get_specific_metadata_columns_string()
        sql_str = (
            f'INSERT INTO "{self.table_name}" ("{self.content_column}", '
            f'"{self.metadata_column}", '
            f'"{self.vector_column}"{specific_metadata_columns_string}) '
            f"VALUES (?, ?, ?"
            f"{', ?' * len(self.specific_metadata_columns)});"
        )

        # Insert data into the table
        cur = self.connection.cursor()
        try:
            cur.executemany(sql_str, sql_params)
        finally:
            cur.close()
        return []

    def _add_texts_using_internal_embedding(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """
        Add texts using SAP HANA's internal embedding function.
        
        This method inserts documents into the vector store and generates embeddings
        directly in the database using HANA's VECTOR_EMBEDDING function. This approach
        offers significant performance advantages over generating embeddings in Python
        and then inserting them, especially for large document sets.
        
        The method constructs a parameterized SQL INSERT statement that:
        1. Inserts document text into the content column
        2. Inserts serialized metadata as JSON into the metadata column
        3. Calls VECTOR_EMBEDDING directly in the database to generate and store embeddings
        4. Optionally extracts specific metadata fields into dedicated columns
        
        Args:
            texts: Iterable of strings/text to add to the vectorstore.
            metadatas: Optional list of metadata dictionaries, one for each text.
            embeddings: Ignored for internal embeddings (embeddings are generated in the database).
            **kwargs: Additional arguments (not used).
            
        Returns:
            list[str]: Empty list (IDs are managed by the database).
            
        Note:
            This method is automatically called by add_texts when the embedding instance
            is of type HanaInternalEmbeddings.
        """
        # Prepare parameters for each document
        sql_params = []
        for i, text in enumerate(texts):
            # Get metadata for this document
            metadata = metadatas[i] if metadatas else {}
            
            # Extract specific metadata fields that should go into dedicated columns
            metadata, extracted_special_metadata = self._split_off_special_metadata(
                metadata
            )
            
            # Prepare parameters for SQL query
            parameters = {
                "content": text,  # Document text
                "metadata": json.dumps(
                    HanaDB._sanitize_metadata_keys(metadata)
                ),  # Serialized metadata JSON
                "model_version": self.internal_embedding_model_id,  # Embedding model ID
            }
            
            # Add specific metadata columns as separate parameters
            parameters.update(
                {
                    col: value
                    for col, value in zip(
                        self.specific_metadata_columns, extracted_special_metadata
                    )
                }
            )
            
            sql_params.append(parameters)

        # Build the SQL INSERT statement
        
        # Create parameter placeholders for specific metadata columns
        specific_metadata_str = ", ".join(
            f":{col}" for col in self.specific_metadata_columns
        )
        
        # Build the column list including specific metadata columns
        specific_metadata_columns_string = self._get_specific_metadata_columns_string()

        # Prepare the VECTOR_EMBEDDING SQL expression
        vector_embedding_sql = "VECTOR_EMBEDDING(:content, 'DOCUMENT', :model_version)"
        
        # Apply type conversion if needed (e.g., convert to HALF_VECTOR if that's the column type)
        vector_embedding_sql = self._convert_vector_embedding_to_column_type(
            vector_embedding_sql
        )

        # Construct the full INSERT statement
        sql_str = (
            f'INSERT INTO "{self.table_name}" ("{self.content_column}", '
            f'"{self.metadata_column}", '
            f'"{self.vector_column}"{specific_metadata_columns_string}) '
            f"VALUES (:content, :metadata, {vector_embedding_sql} "
            f"{(', ' + specific_metadata_str) if specific_metadata_str else ''});"
        )

        # Execute the INSERT statement for all documents
        cur = self.connection.cursor()
        try:
            cur.executemany(sql_str, sql_params)
        finally:
            cur.close()
        
        # Return empty list as IDs are managed by the database
        return []

    def _get_specific_metadata_columns_string(self) -> str:
        """
        Helper function to generate the specific metadata columns as a SQL string.
        Returns:
            str: SQL string for specific metadata columns.
        """
        if not self.specific_metadata_columns:
            return ""
        return ', "' + '", "'.join(self.specific_metadata_columns) + '"'

    @classmethod
    def from_texts(  # type: ignore[no-untyped-def, override]
        cls: Type[HanaDB],
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        connection: dbapi.Connection = None,
        distance_strategy: DistanceStrategy = default_distance_strategy,
        table_name: str = default_table_name,
        content_column: str = default_content_column,
        metadata_column: str = default_metadata_column,
        vector_column: str = default_vector_column,
        vector_column_length: int = default_vector_column_length,
        vector_column_type: str = default_vector_column_type,
        *,
        specific_metadata_columns: Optional[list[str]] = None,
    ):
        """Create a HanaDB instance from raw documents.
        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates a table if it does not yet exist.
            3. Adds the documents to the table.
        This is intended to be a quick way to get started.
        """

        instance = cls(
            connection=connection,
            embedding=embedding,
            distance_strategy=distance_strategy,
            table_name=table_name,
            content_column=content_column,
            metadata_column=metadata_column,
            vector_column=vector_column,
            vector_column_length=vector_column_length,  # -1 means dynamic length
            vector_column_type=vector_column_type,
            specific_metadata_columns=specific_metadata_columns,
        )
        instance.add_texts(texts, metadatas)
        return instance

    def similarity_search(  # type: ignore[override]
        self, query: str, k: int = 4, filter: Optional[dict] = None
    ) -> list[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: A dictionary of metadata fields and values to filter by.
                    Defaults to None.

        Returns:
            List of Documents most similar to the query
        """
        docs_and_scores = self.similarity_search_with_score(
            query=query, k=k, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[dict] = None
    ) -> list[tuple[Document, float]]:
        """Return documents and score values most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: A dictionary of metadata fields and values to filter by.
                    Defaults to None.

        Returns:
            List of tuples (containing a Document and a score) that are
            most similar to the query
        """

        if self.use_internal_embeddings:
            # Internal embeddings: pass the query directly
            whole_result = self.similarity_search_with_score_and_vector_by_query(
                query=query, k=k, filter=filter
            )
        else:
            # External embeddings: generate embedding from the query
            embedding = self.embedding.embed_query(query)
            whole_result = self.similarity_search_with_score_and_vector_by_vector(
                embedding=embedding, k=k, filter=filter
            )

        return [(result_item[0], result_item[1]) for result_item in whole_result]

    def similarity_search_with_score_and_vector_by_vector(
        self, embedding: list[float], k: int = 4, filter: Optional[dict] = None
    ) -> list[tuple[Document, float, list[float]]]:
        """Return docs most similar to the given embedding.

        Args:
            embedding: Pre-computed embedding vector to search with.
            k: Number of Documents to return. Defaults to 4.
            filter: A dictionary of metadata fields and values to filter by.
                    Defaults to None.

        Returns:
            List of tuples, each containing:
            - Document: The matched document with its content and metadata
            - float: The similarity score
            - list[float]: The document's embedding vector
        """
        # Use the appropriate vector conversion function
        embedding_expr = self._convert_to_target_vector_type(expr=str(embedding))

        return self._similarity_search_with_score_and_vector(
            embedding_expr, k=k, filter=filter
        )

    def similarity_search_with_score_and_vector_by_query(
        self, query: str, k: int = 4, filter: Optional[dict] = None
    ) -> list[tuple[Document, float, list[float]]]:
        """
        Return docs most similar to the given query using SAP HANA's internal embedding functionality.

        This method leverages SAP HANA's native VECTOR_EMBEDDING function to generate
        the query embedding directly in the database, which provides several advantages:
        
        1. Performance: Embedding generation happens within the database, reducing data transfer
        2. Consistency: The same embedding model is used for both documents and queries
        3. Efficiency: Avoids loading embedding models in the application server
        4. Scalability: Can leverage SAP HANA's distributed computing resources
        
        To use this method, the embedding instance provided during initialization must
        be of type HanaInternalEmbeddings. This design pattern ensures that all embedding
        operations are consistently delegated to the database.

        Args:
            query: Text to look up documents similar to. This will be passed to
                  HANA's VECTOR_EMBEDDING function for embedding generation.
            k: Number of Documents to return. Defaults to 4.
            filter: A dictionary of metadata fields and values to filter by.
                   Defaults to None. Supports nested filtering with logical operators
                   and various comparison operators for more precise results.

        Returns:
            List of tuples, each containing:
            - Document: The matched document with its content and metadata
            - float: The similarity score (higher is better for cosine similarity)
            - list[float]: The document's embedding vector
            
        Raises:
            TypeError: If self.embedding is not an instance of HanaInternalEmbeddings
            
        Example:
            ```python
            # Create vectorstore with internal embeddings
            from langchain_hana import HanaVectorStore
            from langchain_hana.embeddings import HanaInternalEmbeddings
            
            vectorstore = HanaVectorStore(
                connection=conn,
                embedding=HanaInternalEmbeddings(internal_embedding_model_id="SAP_NEB.20240715"),
                table_name="MY_VECTORS"
            )
            
            # Search with query text - embedding happens in the database
            results = vectorstore.similarity_search_with_score_and_vector_by_query(
                query="What is SAP HANA Cloud?",
                k=5,
                filter={"category": "database"}
            )
            
            # Process results
            for doc, score, vector in results:
                print(f"Content: {doc.page_content[:50]}...")
                print(f"Score: {score}")
                print(f"Vector dimension: {len(vector)}")
                print(f"Metadata: {doc.metadata}")
                print()
            ```
        """
        # Check if the embedding instance is of the correct type
        if not isinstance(self.embedding, HanaInternalEmbeddings):
            raise TypeError(
                "self.embedding must be an instance of "
                "HanaInternalEmbeddings to use the method "
                "similarity_search_with_score_and_vector_by_query"
            )

        # Prepare the SQL expression for VECTOR_EMBEDDING
        embedding_expr = "VECTOR_EMBEDDING(?, 'QUERY', ?)"
        
        # Wrap VECTOR_EMBEDDING with vector type conversion if needed
        # (e.g., if vector_column_type is HALF_VECTOR, convert the REAL_VECTOR output
        # of VECTOR_EMBEDDING to HALF_VECTOR using TO_HALF_VECTOR)
        vector_embedding_sql = self._convert_vector_embedding_to_column_type(
            embedding_expr
        )

        # Parameters for the VECTOR_EMBEDDING function
        vector_embedding_params = [query, self.internal_embedding_model_id]
        
        # Execute the similarity search using the database-generated embedding
        return self._similarity_search_with_score_and_vector(
            vector_embedding_sql,
            vector_embedding_params=vector_embedding_params,
            k=k,
            filter=filter,
        )

    def _similarity_search_with_score_and_vector(
        self,
        embedding_expr: str,
        k: int = 4,
        filter: Optional[dict] = None,
        vector_embedding_params: Optional[list[str]] = None,
    ) -> list[tuple[Document, float, list[float]]]:
        """Perform similarity search and return documents with scores and vectors.

        Args:
            embedding_expr: SQL expression that generates or represents the embedding
                vector. For internal embeddings, this would be
                "VECTOR_EMBEDDING(?, 'QUERY', ?)",
                where ? will be replaced by parameters from query_params.
                For external embeddings, this would be "TO_REAL_VECTOR('...')".
            k: Number of documents to return. Defaults to 4.
            filter: Optional dictionary of metadata fields and values to filter by.
            query_params: Optional parameters for the embedding_expr SQL expression.
                For VECTOR_EMBEDDING function: [text, model_id] for the placeholders.
                For TO_REAL_VECTOR: None as the vector is included in the expression.

        Returns:
            List of tuples, each containing:
            - Document: The matched document with its content and metadata
            - float: The similarity score
            - list[float]: The document's embedding vector
        """
        result = []
        k = HanaDB._sanitize_int(k)
        distance_func_name = HANA_DISTANCE_FUNCTION[self.distance_strategy][0]

        # Generate metadata projection for filtered results
        projected_metadata_columns = self._extract_keyword_search_columns(filter)
        metadata_projection = ""
        if projected_metadata_columns:
            metadata_projection = self._create_metadata_projection(
                projected_metadata_columns
            )
        if metadata_projection:
            from_clause = INTERMEDIATE_TABLE_NAME
        else:
            from_clause = f'"{self.table_name}"'

        sql_str = (
            f"{metadata_projection} "
            f"SELECT TOP {k}"
            f'  "{self.content_column}", '  # row[0]
            f'  "{self.metadata_column}", '  # row[1]
            f'  "{self.vector_column}", '  # row[2]
            f'  {distance_func_name}("{self.vector_column}", '
            f"  {embedding_expr}) AS CS "  # row[3]
            f"FROM {from_clause}"
        )
        parameters = []
        if vector_embedding_params:
            parameters += vector_embedding_params
        where_clause, where_parameters = CreateWhereClause(self)(filter)
        if where_clause:
            sql_str += f" {where_clause}"
            parameters += where_parameters
        sql_str += f" order by CS {HANA_DISTANCE_FUNCTION[self.distance_strategy][1]}"

        try:
            cur = self.connection.cursor()
            if parameters:
                cur.execute(sql_str, parameters)
            else:
                cur.execute(sql_str)
            if cur.has_result_set():
                rows = cur.fetchall()
                for row in rows:
                    js = json.loads(row[1])
                    doc = Document(page_content=row[0], metadata=js)
                    result_vector = self._deserialize_binary_format(row[2])
                    result.append((doc, row[3], result_vector))
        except dbapi.Error:
            logger.error(f"SQL Statement: {sql_str}")
            logger.error(f"Parameters: {parameters}")
            raise
        finally:
            cur.close()
        return result

    def _extract_keyword_search_columns(
        self, filter: Optional[dict] = None
    ) -> list[str]:
        """
        Extract metadata columns used with `$contains` in the filter.
        Scans the filter to find unspecific metadata columns used
        with the `$contains` operator.
        Args:
            filter: A dictionary of filter criteria.
        Returns:
            list of metadata column names for keyword searches.
        Example:
            filter = {"$or": [
                {"title": {"$contains": "barbie"}},
                {"VEC_TEXT": {"$contains": "fred"}}]}
            Result: ["title"]
        """
        keyword_columns: set[str] = set()
        self._recurse_filters(keyword_columns, filter)
        return list(keyword_columns)

    def _recurse_filters(
        self,
        keyword_columns: set[str],
        filter_obj: Optional[dict[Any, Any]],
        parent_key: Optional[str] = None,
    ) -> None:
        """
        Recursively process the filter dictionary
        to find metadata columns used with `$contains`.
        """
        if isinstance(filter_obj, dict):
            for key, value in filter_obj.items():
                if key == CONTAINS_OPERATOR:
                    # Add the parent key as it's the metadata column being filtered
                    if parent_key and not (
                        parent_key == self.content_column
                        or parent_key in self.specific_metadata_columns
                    ):
                        keyword_columns.add(parent_key)
                elif key in LOGICAL_OPERATORS_TO_SQL:  # Handle logical operators
                    for subfilter in value:
                        self._recurse_filters(keyword_columns, subfilter)
                else:
                    self._recurse_filters(keyword_columns, value, parent_key=key)

    def _create_metadata_projection(self, projected_metadata_columns: list[str]) -> str:
        """
        Generate a SQL `WITH` clause to project metadata columns for keyword search.
        Args:
            projected_metadata_columns: List of metadata column names for projection.
        Returns:
            A SQL `WITH` clause string.
        Example:
            Input: ["title", "author"]
            Output:
            WITH intermediate_result AS (
                SELECT *,
                JSON_VALUE(metadata_column, '$.title') AS "title",
                JSON_VALUE(metadata_column, '$.author') AS "author"
                FROM "table_name"
            )
        """

        metadata_columns = [
            (
                f"JSON_VALUE({self.metadata_column}, '$.{HanaDB._sanitize_name(col)}') "
                f'AS "{HanaDB._sanitize_name(col)}"'
            )
            for col in projected_metadata_columns
        ]
        return (
            f"WITH {INTERMEDIATE_TABLE_NAME} AS ("
            f"SELECT *, {', '.join(metadata_columns)} "
            f"FROM \"{self.table_name}\")"
        )

    def similarity_search_by_vector(  # type: ignore[override]
        self, embedding: list[float], k: int = 4, filter: Optional[dict] = None
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: A dictionary of metadata fields and values to filter by.
                    Defaults to None.

        Returns:
            List of Documents most similar to the query vector.
        """
        whole_result = self.similarity_search_with_score_and_vector_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        docs_and_scores = [
            (result_item[0], result_item[1]) for result_item in whole_result
        ]
        return [doc for doc, _ in docs_and_scores]

    def delete(  # type: ignore[override]
        self, ids: Optional[list[str]] = None, filter: Optional[dict] = None
    ) -> Optional[bool]:
        """Delete entries by filter with metadata values

        Args:
            ids: Deletion with ids is not supported! A ValueError will be raised.
            filter: A dictionary of metadata fields and values to filter by.
                    An empty filter ({}) will delete all entries in the table.

        Returns:
            Optional[bool]: True, if deletion is technically successful.
            Deletion of zero entries, due to non-matching filters is a success.
        """

        if ids is not None:
            raise ValueError("Deletion via ids is not supported")

        if filter is None:
            raise ValueError("Parameter 'filter' is required when calling 'delete'")

        where_clause, parameters = CreateWhereClause(self)(filter)
        sql_str = f'DELETE FROM "{self.table_name}" {where_clause}'

        try:
            cur = self.connection.cursor()
            cur.execute(sql_str, parameters)
        finally:
            cur.close()

        return True

    async def adelete(  # type: ignore[override]
        self, ids: Optional[list[str]] = None, filter: Optional[dict] = None
    ) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        return await run_in_executor(None, self.delete, ids=ids, filter=filter)

    def _embed_query_hana_internal(self, query: str) -> list[float]:
        """
        Generate query embedding using SAP HANA's internal embedding engine.
        
        This method executes a SQL query that calls HANA's VECTOR_EMBEDDING function
        to generate an embedding vector for the query text. The embedding is generated
        entirely within the database, which provides performance and consistency benefits.
        
        Key advantages of this approach:
        1. Uses the same embedding model for queries as for documents
        2. Avoids loading embedding models in the application server
        3. Leverages SAP HANA's optimized implementation
        4. Ensures consistent vector representations
        
        Args:
            query: The text to generate an embedding for
            
        Returns:
            list[float]: The embedding vector as a list of floats
            
        Raises:
            ValueError: If no result is returned by the database
            Various database exceptions: If there are issues with the database connection
                                        or the VECTOR_EMBEDDING function
        
        Note:
            This method is used internally by max_marginal_relevance_search when using
            HanaInternalEmbeddings, and can also be used by application code that needs
            direct access to the embedding vectors.
        """
        # Prepare the VECTOR_EMBEDDING SQL expression
        vector_embedding_sql = "VECTOR_EMBEDDING(:content, 'QUERY', :model_version)"
        
        # Apply any necessary type conversion (e.g., to HALF_VECTOR if needed)
        vector_embedding_sql = self._convert_vector_embedding_to_column_type(
            vector_embedding_sql
        )
        
        # Construct the full SQL query
        # We use sys.DUMMY to execute a simple query that doesn't need a table
        sql_str = f"SELECT {vector_embedding_sql} FROM sys.DUMMY;"
        
        # Execute the query
        cur = self.connection.cursor()
        try:
            cur.execute(
                sql_str,
                content=query,
                model_version=self.internal_embedding_model_id,
            )
            
            # Process the result
            if cur.has_result_set():
                res = cur.fetchall()
                # Convert the binary vector format to a list of floats
                return self._deserialize_binary_format(res[0][0])
            else:
                raise ValueError("No result set returned for query embedding.")
        finally:
            cur.close()

    def max_marginal_relevance_search(  # type: ignore[override]
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: search query text.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        if not self.use_internal_embeddings:
            embedding = self.embedding.embed_query(query)
        else:  # generates embedding using the internal embedding function of HanaDb
            embedding = self._embed_query_hana_internal(query)

        return self.max_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
        )

    def _parse_float_array_from_string(array_as_string: str) -> list[float]:  # type: ignore[misc]
        array_wo_brackets = array_as_string[1:-1]
        return [float(x) for x in array_wo_brackets.split(",")]

    def max_marginal_relevance_search_by_vector(  # type: ignore[override]
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        whole_result = self.similarity_search_with_score_and_vector_by_vector(
            embedding=embedding, k=fetch_k, filter=filter
        )
        embeddings = [result_item[2] for result_item in whole_result]
        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(embedding), embeddings, lambda_mult=lambda_mult, k=k
        )

        return [whole_result[i][0] for i in mmr_doc_indexes]

    async def amax_marginal_relevance_search_by_vector(  # type: ignore[override]
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance."""
        return await run_in_executor(
            None,
            self.max_marginal_relevance_search_by_vector,
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )

    @staticmethod
    def _cosine_relevance_score_fn(distance: float) -> float:
        return distance

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.

        Vectorstores should define their own selection based method of relevance.
        """
        if self.distance_strategy == DistanceStrategy.COSINE:
            return HanaDB._cosine_relevance_score_fn
        elif self.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            return HanaDB._euclidean_relevance_score_fn
        else:
            raise ValueError(
                "Unsupported distance_strategy: {}".format(self.distance_strategy)
            )
