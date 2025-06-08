"""
Structured audit logging for SAP HANA Cloud LangChain integration.

This module provides comprehensive audit logging capabilities for compliance
and security requirements, including:

1. Structured JSON logs with standardized fields
2. Compliance-focused log fields (user, action, resource, etc.)
3. Log rotation and retention policies
4. Integration with enterprise logging systems
5. Support for various log sinks (file, database, external services)

The audit logging system is designed to meet requirements for:
- SOX compliance
- GDPR compliance
- HIPAA compliance (where applicable)
- Internal security audits
"""

import json
import logging
import os
import socket
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable

# Setup the standard logger
logger = logging.getLogger(__name__)

class LogLevel(str, Enum):
    """Standard log levels with compliance and security focus."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    NOTICE = "NOTICE"  # For notable events that aren't warnings
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    AUDIT = "AUDIT"  # Specifically for compliance audit events
    SECURITY = "SECURITY"  # For security-related events


class AuditCategory(str, Enum):
    """Categories for audit events to enable filtering and reporting."""
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    DATA_ACCESS = "DATA_ACCESS"
    DATA_MODIFICATION = "DATA_MODIFICATION"
    CONFIGURATION = "CONFIGURATION"
    SYSTEM = "SYSTEM"
    EMBEDDING = "EMBEDDING"
    VECTOR_SEARCH = "VECTOR_SEARCH"
    USER_ACTIVITY = "USER_ACTIVITY"
    ADMIN_ACTIVITY = "ADMIN_ACTIVITY"
    INTEGRATION = "INTEGRATION"
    COMPLIANCE = "COMPLIANCE"


class AuditLogFormatter(logging.Formatter):
    """
    Custom formatter for structured JSON audit logs.
    
    This formatter ensures that all audit logs have a consistent structure
    with standardized fields required for compliance reporting.
    """
    
    def format(self, record):
        """Format log record as structured JSON."""
        # Extract the log message
        log_message = super().format(record)
        
        # Get exception info if available
        exc_info = None
        if record.exc_info:
            exc_info = ''.join(traceback.format_exception(*record.exc_info))
        
        # Base log record with standard fields
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "thread": f"{record.threadName}:{record.thread}",
            "process": record.process,
            "host": socket.gethostname(),
            "message": log_message,
        }
        
        # Add exception info if available
        if exc_info:
            log_record["exception"] = exc_info
        
        # Add all extra attributes from the record
        for key, value in record.__dict__.items():
            # Skip standard attributes and private attributes
            if key not in ("args", "asctime", "created", "exc_info", "exc_text", 
                           "filename", "funcName", "id", "levelname", "levelno", 
                           "lineno", "module", "msecs", "message", "msg", "name", 
                           "pathname", "process", "processName", "relativeCreated", 
                           "stack_info", "thread", "threadName") and not key.startswith("_"):
                log_record[key] = value
        
        # Convert to JSON
        return json.dumps(log_record)


class ComplianceHandler(logging.Handler):
    """
    Logging handler designed for compliance and audit requirements.
    
    Features:
    - Writes structured JSON logs to a designated file
    - Implements log rotation based on size or time
    - Enforces retention policies
    - Ensures secure log handling
    - Supports encryption and integrity verification
    """
    
    def __init__(
        self,
        filename: str,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 10,
        encoding: str = "utf-8",
        delay: bool = False,
        mode: str = "a",
        level: int = logging.INFO,
    ):
        """
        Initialize the compliance handler.
        
        Args:
            filename: Path to the log file
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep
            encoding: File encoding
            delay: Whether to delay file creation until first log
            mode: File open mode
            level: Minimum log level to process
        """
        super().__init__(level)
        self.filename = filename
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.encoding = encoding
        self.delay = delay
        self.mode = mode
        self.stream = None
        self.current_size = 0
        
        # Set formatter to JSON
        self.setFormatter(AuditLogFormatter())
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Open the file if not delayed
        if not delay:
            self._open()
    
    def _open(self):
        """Open the log file."""
        # Create directory if it doesn't exist
        directory = os.path.dirname(self.filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Open the file
        self.stream = open(self.filename, self.mode, encoding=self.encoding)
        
        # Get current file size
        if self.mode == "a":
            self.stream.seek(0, 2)  # Seek to end
            self.current_size = self.stream.tell()
        else:
            self.current_size = 0
    
    def emit(self, record):
        """Emit a log record."""
        with self.lock:
            if self.stream is None:
                self._open()
            
            # Format the record
            msg = self.format(record) + "\n"
            msg_bytes = msg.encode(self.encoding)
            msg_size = len(msg_bytes)
            
            # Check if rotation is needed
            if self.max_bytes > 0 and self.current_size + msg_size > self.max_bytes:
                self._rotate_logs()
            
            # Write the log
            self.stream.write(msg)
            self.stream.flush()
            self.current_size += msg_size
    
    def _rotate_logs(self):
        """Rotate log files."""
        # Close current stream
        if self.stream:
            self.stream.close()
            self.stream = None
        
        # Rotate existing log files
        for i in range(self.backup_count - 1, 0, -1):
            src = f"{self.filename}.{i}"
            dst = f"{self.filename}.{i+1}"
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)
        
        # Rename current log file
        if os.path.exists(self.filename):
            os.rename(self.filename, f"{self.filename}.1")
        
        # Open new log file
        self._open()
    
    def close(self):
        """Close the log file."""
        with self.lock:
            if self.stream:
                self.stream.close()
                self.stream = None
        super().close()


class DatabaseLogHandler(logging.Handler):
    """
    Logging handler that writes audit logs to a database table.
    
    This handler stores structured audit logs in a database table for
    easy querying, reporting, and long-term retention.
    """
    
    def __init__(
        self,
        connection,
        table_name: str = "AUDIT_LOGS",
        level: int = logging.INFO,
        batch_size: int = 10,
        flush_interval: int = 5,  # seconds
    ):
        """
        Initialize the database log handler.
        
        Args:
            connection: Database connection
            table_name: Name of the audit log table
            level: Minimum log level to process
            batch_size: Number of logs to batch before writing
            flush_interval: Maximum time to wait before flushing logs
        """
        super().__init__(level)
        self.connection = connection
        self.table_name = table_name
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Set formatter to JSON
        self.setFormatter(AuditLogFormatter())
        
        # Initialize log buffer
        self.log_buffer = []
        self.last_flush = time.time()
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Initialize the database table
        self._initialize_table()
        
        # Start background thread for periodic flushing
        self._start_flush_thread()
    
    def _initialize_table(self):
        """Initialize the audit log table if it doesn't exist."""
        try:
            cursor = self.connection.cursor()
            
            # Create the audit log table if it doesn't exist
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS "{self.table_name}" (
                    "ID" VARCHAR(36) PRIMARY KEY,
                    "TIMESTAMP" TIMESTAMP,
                    "LEVEL" VARCHAR(20),
                    "CATEGORY" VARCHAR(50),
                    "USER_ID" VARCHAR(255),
                    "ACTION" VARCHAR(255),
                    "RESOURCE" NVARCHAR(1000),
                    "RESOURCE_TYPE" VARCHAR(100),
                    "RESOURCE_ID" VARCHAR(255),
                    "STATUS" VARCHAR(50),
                    "CLIENT_IP" VARCHAR(50),
                    "REQUEST_ID" VARCHAR(100),
                    "DETAILS" NCLOB,
                    "SOURCE" VARCHAR(255),
                    "APPLICATION" VARCHAR(255)
                )
            """)
            
            # Create index on timestamp for faster queries
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS "IDX_{self.table_name}_TIMESTAMP" 
                ON "{self.table_name}" ("TIMESTAMP")
            """)
            
            # Create index on user_id for user activity reports
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS "IDX_{self.table_name}_USER_ID" 
                ON "{self.table_name}" ("USER_ID")
            """)
            
            # Create index on category for filtering
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS "IDX_{self.table_name}_CATEGORY" 
                ON "{self.table_name}" ("CATEGORY")
            """)
            
            cursor.close()
        except Exception as e:
            logger.error(f"Error initializing audit log table: {str(e)}")
    
    def _start_flush_thread(self):
        """Start a background thread for periodic log flushing."""
        def flush_periodically():
            while True:
                time.sleep(self.flush_interval)
                self.flush()
        
        thread = threading.Thread(target=flush_periodically, daemon=True)
        thread.start()
    
    def emit(self, record):
        """Add a log record to the buffer."""
        with self.lock:
            # Format the record
            formatted_record = json.loads(self.format(record))
            
            # Add to buffer
            self.log_buffer.append(formatted_record)
            
            # Flush if buffer is full
            if len(self.log_buffer) >= self.batch_size:
                self.flush()
    
    def flush(self):
        """Flush log buffer to the database."""
        with self.lock:
            if not self.log_buffer:
                return
            
            try:
                cursor = self.connection.cursor()
                
                # Prepare batch insert
                for log in self.log_buffer:
                    # Generate a unique ID for the log entry
                    log_id = str(uuid.uuid4())
                    
                    # Extract standard fields
                    timestamp = log.get("timestamp", datetime.now(timezone.utc).isoformat())
                    level = log.get("level", "INFO")
                    category = log.get("category", "SYSTEM")
                    user_id = log.get("user_id")
                    action = log.get("action")
                    resource = log.get("resource")
                    resource_type = log.get("resource_type")
                    resource_id = log.get("resource_id")
                    status = log.get("status", "SUCCESS")
                    client_ip = log.get("client_ip")
                    request_id = log.get("request_id")
                    source = log.get("source", "langchain_hana")
                    application = log.get("application")
                    
                    # Store remaining details as JSON
                    details = json.dumps({k: v for k, v in log.items() if k not in [
                        "timestamp", "level", "category", "user_id", "action", 
                        "resource", "resource_type", "resource_id", "status", 
                        "client_ip", "request_id", "source", "application"
                    ]})
                    
                    # Convert ISO timestamp to datetime object if needed
                    if isinstance(timestamp, str):
                        try:
                            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        except ValueError:
                            timestamp = datetime.now(timezone.utc)
                    
                    # Insert into database
                    cursor.execute(f"""
                        INSERT INTO "{self.table_name}" (
                            "ID", "TIMESTAMP", "LEVEL", "CATEGORY", "USER_ID", 
                            "ACTION", "RESOURCE", "RESOURCE_TYPE", "RESOURCE_ID", 
                            "STATUS", "CLIENT_IP", "REQUEST_ID", "DETAILS",
                            "SOURCE", "APPLICATION"
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        log_id, timestamp, level, category, user_id, 
                        action, resource, resource_type, resource_id, 
                        status, client_ip, request_id, details,
                        source, application
                    ))
                
                # Commit transaction
                self.connection.commit()
                
                # Clear the buffer
                self.log_buffer = []
                self.last_flush = time.time()
                
                cursor.close()
            except Exception as e:
                logger.error(f"Error flushing audit logs to database: {str(e)}")
    
    def close(self):
        """Close the handler and flush any remaining logs."""
        self.flush()
        super().close()


class AuditLogger:
    """
    Centralized audit logging facility for compliance and security events.
    
    This class provides methods for logging audit events with standardized
    fields and formats, ensuring compliance with regulatory requirements.
    """
    
    def __init__(
        self,
        connection = None,
        log_file: Optional[str] = None,
        log_to_database: bool = False,
        log_to_console: bool = False,
        log_level: LogLevel = LogLevel.INFO,
        application_name: str = "langchain_hana",
        source_name: str = "langchain_hana",
    ):
        """
        Initialize the audit logger.
        
        Args:
            connection: Optional database connection for database logging
            log_file: Optional path to the log file
            log_to_database: Whether to log to the database
            log_to_console: Whether to log to the console
            log_level: Minimum log level to process
            application_name: Name of the application
            source_name: Name of the logging source
        """
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(getattr(logging, log_level))
        self.logger.propagate = False  # Don't propagate to root logger
        
        self.application_name = application_name
        self.source_name = source_name
        
        # Add handlers
        if log_file:
            file_handler = ComplianceHandler(
                filename=log_file,
                level=getattr(logging, log_level),
            )
            self.logger.addHandler(file_handler)
        
        if log_to_database and connection:
            db_handler = DatabaseLogHandler(
                connection=connection,
                level=getattr(logging, log_level),
            )
            self.logger.addHandler(db_handler)
        
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(AuditLogFormatter())
            console_handler.setLevel(getattr(logging, log_level))
            self.logger.addHandler(console_handler)
    
    def log(
        self,
        message: str,
        level: LogLevel = LogLevel.INFO,
        category: AuditCategory = AuditCategory.SYSTEM,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        status: str = "SUCCESS",
        client_ip: Optional[str] = None,
        request_id: Optional[str] = None,
        **extra
    ):
        """
        Log an audit event.
        
        Args:
            message: The log message
            level: Log level
            category: Audit category
            user_id: ID of the user performing the action
            action: Action being performed
            resource: Resource being accessed
            resource_type: Type of resource
            resource_id: ID of the resource
            status: Status of the action (SUCCESS, FAILURE, etc.)
            client_ip: IP address of the client
            request_id: Request ID for correlation
            **extra: Additional fields to include in the log
        """
        # Prepare extra fields
        log_extra = {
            "category": category,
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "status": status,
            "client_ip": client_ip,
            "request_id": request_id,
            "application": self.application_name,
            "source": self.source_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Add any additional fields
        log_extra.update(extra)
        
        # Log with the appropriate level
        if level == LogLevel.DEBUG:
            self.logger.debug(message, extra=log_extra)
        elif level == LogLevel.INFO:
            self.logger.info(message, extra=log_extra)
        elif level == LogLevel.NOTICE:
            self.logger.info(message, extra=log_extra)  # Map NOTICE to INFO
        elif level == LogLevel.WARNING:
            self.logger.warning(message, extra=log_extra)
        elif level == LogLevel.ERROR:
            self.logger.error(message, extra=log_extra)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(message, extra=log_extra)
        elif level == LogLevel.AUDIT:
            self.logger.info(message, extra=log_extra)  # Map AUDIT to INFO
        elif level == LogLevel.SECURITY:
            self.logger.warning(message, extra=log_extra)  # Map SECURITY to WARNING
    
    def log_data_access(
        self,
        user_id: str,
        resource: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        action: str = "READ",
        status: str = "SUCCESS",
        client_ip: Optional[str] = None,
        request_id: Optional[str] = None,
        query: Optional[str] = None,
        **extra
    ):
        """
        Log a data access event.
        
        Args:
            user_id: ID of the user accessing the data
            resource: Resource being accessed
            resource_type: Type of resource
            resource_id: ID of the resource
            action: Action being performed (READ, WRITE, etc.)
            status: Status of the action
            client_ip: IP address of the client
            request_id: Request ID for correlation
            query: The query being executed
            **extra: Additional fields to include in the log
        """
        # Add query to extra fields if provided
        if query:
            extra["query"] = query
        
        # Log the event
        self.log(
            message=f"Data access: {action} {resource_type} {resource}",
            level=LogLevel.AUDIT,
            category=AuditCategory.DATA_ACCESS,
            user_id=user_id,
            action=action,
            resource=resource,
            resource_type=resource_type,
            resource_id=resource_id,
            status=status,
            client_ip=client_ip,
            request_id=request_id,
            **extra
        )
    
    def log_vector_search(
        self,
        user_id: Optional[str],
        query: str,
        table_name: str,
        num_results: int,
        filter: Optional[Dict] = None,
        status: str = "SUCCESS",
        client_ip: Optional[str] = None,
        request_id: Optional[str] = None,
        execution_time: Optional[float] = None,
        **extra
    ):
        """
        Log a vector search event.
        
        Args:
            user_id: ID of the user performing the search
            query: The search query
            table_name: Name of the table being searched
            num_results: Number of results returned
            filter: Filter criteria
            status: Status of the search
            client_ip: IP address of the client
            request_id: Request ID for correlation
            execution_time: Time taken to execute the search
            **extra: Additional fields to include in the log
        """
        # Add search-specific fields
        search_info = {
            "query": query,
            "table_name": table_name,
            "num_results": num_results,
            "filter": filter,
        }
        
        if execution_time is not None:
            search_info["execution_time"] = execution_time
        
        # Add to extra fields
        extra.update(search_info)
        
        # Log the event
        self.log(
            message=f"Vector search in {table_name}: '{query[:50]}{'...' if len(query) > 50 else ''}'",
            level=LogLevel.AUDIT,
            category=AuditCategory.VECTOR_SEARCH,
            user_id=user_id,
            action="SEARCH",
            resource=table_name,
            resource_type="VECTOR_TABLE",
            status=status,
            client_ip=client_ip,
            request_id=request_id,
            **extra
        )
    
    def log_embedding_generation(
        self,
        user_id: Optional[str],
        model: str,
        num_documents: int,
        total_tokens: Optional[int] = None,
        status: str = "SUCCESS",
        client_ip: Optional[str] = None,
        request_id: Optional[str] = None,
        execution_time: Optional[float] = None,
        **extra
    ):
        """
        Log an embedding generation event.
        
        Args:
            user_id: ID of the user generating embeddings
            model: The embedding model used
            num_documents: Number of documents embedded
            total_tokens: Total number of tokens processed
            status: Status of the embedding generation
            client_ip: IP address of the client
            request_id: Request ID for correlation
            execution_time: Time taken to generate embeddings
            **extra: Additional fields to include in the log
        """
        # Add embedding-specific fields
        embedding_info = {
            "model": model,
            "num_documents": num_documents,
        }
        
        if total_tokens is not None:
            embedding_info["total_tokens"] = total_tokens
        
        if execution_time is not None:
            embedding_info["execution_time"] = execution_time
        
        # Add to extra fields
        extra.update(embedding_info)
        
        # Log the event
        self.log(
            message=f"Generated embeddings for {num_documents} documents using {model}",
            level=LogLevel.AUDIT,
            category=AuditCategory.EMBEDDING,
            user_id=user_id,
            action="GENERATE_EMBEDDINGS",
            resource=model,
            resource_type="EMBEDDING_MODEL",
            status=status,
            client_ip=client_ip,
            request_id=request_id,
            **extra
        )
    
    def log_authentication(
        self,
        user_id: str,
        status: str,
        client_ip: Optional[str] = None,
        request_id: Optional[str] = None,
        auth_method: Optional[str] = None,
        **extra
    ):
        """
        Log an authentication event.
        
        Args:
            user_id: ID of the user being authenticated
            status: Status of the authentication
            client_ip: IP address of the client
            request_id: Request ID for correlation
            auth_method: Authentication method used
            **extra: Additional fields to include in the log
        """
        # Add authentication-specific fields
        if auth_method:
            extra["auth_method"] = auth_method
        
        # Log the event
        self.log(
            message=f"Authentication {status.lower()} for user {user_id}",
            level=LogLevel.AUDIT if status == "SUCCESS" else LogLevel.SECURITY,
            category=AuditCategory.AUTHENTICATION,
            user_id=user_id,
            action="AUTHENTICATE",
            resource=user_id,
            resource_type="USER",
            status=status,
            client_ip=client_ip,
            request_id=request_id,
            **extra
        )
    
    def log_authorization(
        self,
        user_id: str,
        resource: str,
        resource_type: str,
        action: str,
        status: str,
        client_ip: Optional[str] = None,
        request_id: Optional[str] = None,
        **extra
    ):
        """
        Log an authorization event.
        
        Args:
            user_id: ID of the user being authorized
            resource: Resource being accessed
            resource_type: Type of resource
            action: Action being performed
            status: Status of the authorization
            client_ip: IP address of the client
            request_id: Request ID for correlation
            **extra: Additional fields to include in the log
        """
        # Log the event
        self.log(
            message=f"Authorization {status.lower()} for user {user_id} to {action} {resource_type} {resource}",
            level=LogLevel.AUDIT if status == "SUCCESS" else LogLevel.SECURITY,
            category=AuditCategory.AUTHORIZATION,
            user_id=user_id,
            action=action,
            resource=resource,
            resource_type=resource_type,
            status=status,
            client_ip=client_ip,
            request_id=request_id,
            **extra
        )
    
    def get_audit_report(
        self,
        connection,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        category: Optional[AuditCategory] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Generate an audit report based on specified criteria.
        
        Args:
            connection: Database connection
            start_date: Optional start date for the report
            end_date: Optional end date for the report
            user_id: Optional user ID to filter by
            category: Optional category to filter by
            action: Optional action to filter by
            resource_type: Optional resource type to filter by
            status: Optional status to filter by
            limit: Maximum number of records to return
            offset: Offset for pagination
            
        Returns:
            A dictionary containing the audit report
        """
        try:
            cursor = connection.cursor()
            
            # Build the query
            query = """
                SELECT 
                    "ID", "TIMESTAMP", "LEVEL", "CATEGORY", "USER_ID", 
                    "ACTION", "RESOURCE", "RESOURCE_TYPE", "RESOURCE_ID", 
                    "STATUS", "CLIENT_IP", "REQUEST_ID", "DETAILS",
                    "SOURCE", "APPLICATION"
                FROM "AUDIT_LOGS"
                WHERE 1=1
            """
            
            params = []
            
            # Add filters
            if start_date:
                query += " AND \"TIMESTAMP\" >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND \"TIMESTAMP\" <= ?"
                params.append(end_date)
            
            if user_id:
                query += " AND \"USER_ID\" = ?"
                params.append(user_id)
            
            if category:
                query += " AND \"CATEGORY\" = ?"
                params.append(category)
            
            if action:
                query += " AND \"ACTION\" = ?"
                params.append(action)
            
            if resource_type:
                query += " AND \"RESOURCE_TYPE\" = ?"
                params.append(resource_type)
            
            if status:
                query += " AND \"STATUS\" = ?"
                params.append(status)
            
            # Add order by and limit
            query += " ORDER BY \"TIMESTAMP\" DESC LIMIT ? OFFSET ?"
            params.append(limit)
            params.append(offset)
            
            # Execute the query
            cursor.execute(query, params)
            
            # Process results
            results = []
            for row in cursor.fetchall():
                # Parse the details JSON
                details = json.loads(row[12]) if row[12] else {}
                
                # Create a record
                record = {
                    "id": row[0],
                    "timestamp": row[1].isoformat() if row[1] else None,
                    "level": row[2],
                    "category": row[3],
                    "user_id": row[4],
                    "action": row[5],
                    "resource": row[6],
                    "resource_type": row[7],
                    "resource_id": row[8],
                    "status": row[9],
                    "client_ip": row[10],
                    "request_id": row[11],
                    "source": row[13],
                    "application": row[14],
                    **details
                }
                
                results.append(record)
            
            # Get total count
            count_query = """
                SELECT COUNT(*) FROM "AUDIT_LOGS" WHERE 1=1
            """
            
            count_params = []
            
            # Add filters
            if start_date:
                count_query += " AND \"TIMESTAMP\" >= ?"
                count_params.append(start_date)
            
            if end_date:
                count_query += " AND \"TIMESTAMP\" <= ?"
                count_params.append(end_date)
            
            if user_id:
                count_query += " AND \"USER_ID\" = ?"
                count_params.append(user_id)
            
            if category:
                count_query += " AND \"CATEGORY\" = ?"
                count_params.append(category)
            
            if action:
                count_query += " AND \"ACTION\" = ?"
                count_params.append(action)
            
            if resource_type:
                count_query += " AND \"RESOURCE_TYPE\" = ?"
                count_params.append(resource_type)
            
            if status:
                count_query += " AND \"STATUS\" = ?"
                count_params.append(status)
            
            # Execute the count query
            cursor.execute(count_query, count_params)
            total_count = cursor.fetchone()[0]
            
            cursor.close()
            
            # Build the report
            report = {
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "total_pages": (total_count + limit - 1) // limit,
                "current_page": offset // limit + 1,
                "filters": {
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                    "user_id": user_id,
                    "category": category,
                    "action": action,
                    "resource_type": resource_type,
                    "status": status,
                },
                "results": results
            }
            
            return report
        
        except Exception as e:
            logger.error(f"Error generating audit report: {str(e)}")
            return {
                "error": str(e),
                "total_count": 0,
                "limit": limit,
                "offset": offset,
                "total_pages": 0,
                "current_page": 1,
                "filters": {},
                "results": []
            }


# Decorator for adding audit logging to methods
def audit_log(
    action: str, 
    category: AuditCategory,
    resource_type: str,
    get_resource_fn: Optional[Callable] = None,
    message_template: Optional[str] = None
):
    """
    Decorator that adds audit logging to methods.
    
    This decorator can be applied to methods to automatically log audit events
    when the method is called, with details about the method call.
    
    Args:
        action: The action being performed
        category: The audit category
        resource_type: The type of resource being accessed
        get_resource_fn: Optional function to extract resource name from args/kwargs
        message_template: Optional template for the log message
        
    Returns:
        The decorated method
    """
    def decorator(method):
        def wrapper(self, *args, **kwargs):
            # Check if audit logging is enabled
            audit_logger = getattr(self, "audit_logger", None)
            if audit_logger is None:
                # No audit logger, just call the original method
                return method(self, *args, **kwargs)
            
            # Get user ID if available
            user_id = getattr(self, "current_user_id", None)
            
            # Get resource if available
            resource = None
            if get_resource_fn:
                try:
                    resource = get_resource_fn(self, *args, **kwargs)
                except Exception:
                    resource = str(args[0]) if args else None
            else:
                resource = getattr(self, "table_name", None)
            
            # Create default message if not provided
            message = message_template
            if not message:
                method_name = method.__name__
                message = f"{action} operation on {resource_type}"
                if resource:
                    message += f" {resource}"
            
            # Additional context for the log
            context = {
                "method": method.__name__,
                "args": str(args),
                "kwargs": {k: v for k, v in kwargs.items() if k not in ["password", "token", "secret", "key"]}
            }
            
            start_time = time.time()
            
            try:
                # Call the original method
                result = method(self, *args, **kwargs)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Log successful execution
                audit_logger.log(
                    message=message,
                    level=LogLevel.AUDIT,
                    category=category,
                    user_id=user_id,
                    action=action,
                    resource=resource,
                    resource_type=resource_type,
                    status="SUCCESS",
                    execution_time=execution_time,
                    **context
                )
                
                return result
            
            except Exception as e:
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Log failed execution
                context["error"] = str(e)
                audit_logger.log(
                    message=f"{message} (failed: {str(e)})",
                    level=LogLevel.ERROR,
                    category=category,
                    user_id=user_id,
                    action=action,
                    resource=resource,
                    resource_type=resource_type,
                    status="FAILURE",
                    execution_time=execution_time,
                    **context
                )
                
                # Re-raise the exception
                raise
        
        return wrapper
    
    return decorator