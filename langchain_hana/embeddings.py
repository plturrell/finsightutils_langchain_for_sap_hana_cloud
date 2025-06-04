from langchain_core.embeddings import Embeddings


class HanaInternalEmbeddings(Embeddings):
    """
    A specialized embeddings class designed to work with SAP HANA Cloud's internal embedding functionality.
    
    Unlike standard embedding classes that perform embedding generation in Python,
    this class delegates embedding generation to SAP HANA's native VECTOR_EMBEDDING function.
    
    This architecture provides several advantages:
    1. Performance: Embeddings are generated directly in the database, reducing data transfer overhead
    2. Resource efficiency: Database CPU/GPU resources are used instead of application resources
    3. Consistency: Embeddings are generated using the same model in both search and insertion operations
    4. Scalability: Can leverage SAP HANA's distributed computing capabilities for large workloads
    
    The class intentionally raises NotImplementedError for standard embedding methods to ensure
    that all embedding operations are performed by the database engine via SQL queries.
    
    Example:
        ```python
        from langchain_hana import HanaVectorStore
        from langchain_hana.embeddings import HanaInternalEmbeddings
        
        # Use SAP HANA's internal embedding model
        embeddings = HanaInternalEmbeddings(internal_embedding_model_id="SAP_NEB.20240715")
        
        # Create vector store with internal embeddings
        vector_store = HanaVectorStore(
            connection=conn,
            embedding=embeddings,
            table_name="MY_VECTORS"
        )
        
        # When similarity_search is called, embedding generation will happen in the database
        results = vector_store.similarity_search("What is SAP HANA?")
        ```
    """

    def __init__(self, internal_embedding_model_id: str):
        """
        Initialize the HanaInternalEmbeddings instance.
        
        Args:
            internal_embedding_model_id (str): The ID of the internal embedding model
                                               used by the HANA database. This should match a
                                               valid model ID in your SAP HANA Cloud instance,
                                               such as "SAP_NEB.20240715".
                                               
        Notes:
            - The model_id is passed to the VECTOR_EMBEDDING function in HANA SQL queries
            - The validity of the model_id is checked when the first query is executed
            - Available models depend on your SAP HANA Cloud version and configuration
        """
        self.model_id = internal_embedding_model_id

    def embed_query(self, text: str) -> list[float]:
        """
        Override the embed_query method to raise an error.
        
        This method is intentionally not implemented for HanaInternalEmbeddings because
        query embedding generation is delegated to SAP HANA's VECTOR_EMBEDDING function
        and executed directly in the database through SQL queries.
        
        When using HanaInternalEmbeddings with HanaVectorStore, the vectorstore will
        automatically generate embeddings through SQL by calling VECTOR_EMBEDDING in
        similarity_search_with_score_and_vector_by_query.
        
        Raises:
            NotImplementedError: Always raised to enforce database-side embedding generation.
        """
        raise NotImplementedError(
            "Internal embeddings cannot be used externally. "
            "Use HANA's internal embedding functionality instead."
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Override the embed_documents method to raise an error.
        
        This method is intentionally not implemented for HanaInternalEmbeddings because
        document embedding generation is delegated to SAP HANA's VECTOR_EMBEDDING function
        and executed directly in the database through SQL queries.
        
        When using HanaInternalEmbeddings with HanaVectorStore, the vectorstore's add_texts
        method will automatically generate embeddings through SQL by calling VECTOR_EMBEDDING
        in _add_texts_using_internal_embedding.
        
        Raises:
            NotImplementedError: Always raised to enforce database-side embedding generation.
        """
        raise NotImplementedError(
            "Internal embeddings cannot be used externally. "
            "Use HANA's internal embedding functionality instead."
        )

    def get_model_id(self) -> str:
        """
        Retrieve the internal embedding model ID.
        
        This method is used by HanaVectorStore to get the model ID that should be
        passed to the VECTOR_EMBEDDING function in SQL queries.
        
        Returns:
            str: The ID of the internal embedding model (e.g., "SAP_NEB.20240715").
            
        Notes:
            - This model ID must match one of the embedding models available in your
              SAP HANA Cloud instance
            - The model ID is validated when the first query is executed
            - Available models may vary depending on your SAP HANA Cloud version and configuration
        """
        return self.model_id
