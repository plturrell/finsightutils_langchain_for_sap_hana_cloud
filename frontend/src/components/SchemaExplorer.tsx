import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Collapse,
  CircularProgress,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Grid,
  Alert,
  Card,
  CardContent,
  alpha,
  useTheme,
} from '@mui/material';
import {
  Folder as FolderIcon,
  FolderOpen as FolderOpenIcon,
  Storage as StorageIcon,
  ViewColumn as ViewColumnIcon,
  KeyboardArrowDown as ExpandMoreIcon,
  KeyboardArrowUp as ExpandLessIcon,
  Refresh as RefreshIcon,
  Search as SearchIcon,
  InfoOutlined as InfoIcon,
  DataObject as DataObjectIcon,
  AutoAwesome as AutoAwesomeIcon,
} from '@mui/icons-material';
import HumanText from './HumanText';
import apiClient from '../api/client';

// Types for HANA schema explorer
interface SchemaMetadata {
  schema_name: string;
  table_count: number;
  view_count: number;
  created: string;
  owner: string;
}

interface TableMetadata {
  table_name: string;
  schema_name: string;
  column_count: number;
  row_count: number;
  size_mb: number;
  created: string;
  is_column_table: boolean;
}

interface ColumnMetadata {
  column_name: string;
  data_type: string;
  length: number;
  nullable: boolean;
  position: number;
  is_primary_key: boolean;
}

interface TableSample {
  columns: string[];
  rows: any[][];
}

// Props for the SchemaExplorer component
interface SchemaExplorerProps {
  onTableSelect?: (schema: string, table: string) => void;
  initialSchema?: string;
}

const SchemaExplorer: React.FC<SchemaExplorerProps> = ({
  onTableSelect,
  initialSchema = 'SYSTEM',
}) => {
  const theme = useTheme();

  // State for schema and table exploration
  const [schemas, setSchemas] = useState<SchemaMetadata[]>([]);
  const [expandedSchema, setExpandedSchema] = useState<string | null>(initialSchema);
  const [tables, setTables] = useState<Record<string, TableMetadata[]>>({});
  const [selectedTable, setSelectedTable] = useState<{schema: string, table: string} | null>(null);
  const [columns, setColumns] = useState<ColumnMetadata[]>([]);
  const [tableSample, setTableSample] = useState<TableSample | null>(null);
  
  // State for loading indicators
  const [isLoadingSchemas, setIsLoadingSchemas] = useState<boolean>(false);
  const [isLoadingTables, setIsLoadingTables] = useState<boolean>(false);
  const [isLoadingColumns, setIsLoadingColumns] = useState<boolean>(false);
  const [isLoadingSample, setIsLoadingSample] = useState<boolean>(false);
  
  // State for error messages
  const [error, setError] = useState<string | null>(null);
  
  // State for search
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [searchResults, setSearchResults] = useState<{schema: string, table: string}[]>([]);
  
  // Load schemas
  const fetchSchemas = async () => {
    try {
      setIsLoadingSchemas(true);
      setError(null);
      
      // In a real implementation, this would call an API endpoint
      // For this demo, we'll use a mock response
      
      // Mock API call
      // const response = await apiClient.get('/api/schemas');
      // const data = response.data;
      
      // Mock data
      const mockSchemas: SchemaMetadata[] = [
        {
          schema_name: 'SYSTEM',
          table_count: 412,
          view_count: 89,
          created: '2023-01-15',
          owner: 'SYSTEM',
        },
        {
          schema_name: 'SYS',
          table_count: 156,
          view_count: 42,
          created: '2023-01-15',
          owner: 'SYSTEM',
        },
        {
          schema_name: 'CUSTOMER_DATA',
          table_count: 28,
          view_count: 12,
          created: '2023-03-22',
          owner: 'DBADMIN',
        },
        {
          schema_name: 'SALES',
          table_count: 37,
          view_count: 15,
          created: '2023-04-10',
          owner: 'DBADMIN',
        },
        {
          schema_name: 'PRODUCTS',
          table_count: 14,
          view_count: 8,
          created: '2023-04-12',
          owner: 'DBADMIN',
        },
      ];
      
      setSchemas(mockSchemas);
      
      // If there's an initial schema, load its tables
      if (initialSchema) {
        fetchTables(initialSchema);
      }
    } catch (err: any) {
      console.error('Error fetching schemas:', err);
      setError(err.response?.data?.message || 'Failed to fetch schemas');
    } finally {
      setIsLoadingSchemas(false);
    }
  };
  
  // Load tables for a schema
  const fetchTables = async (schemaName: string) => {
    try {
      setIsLoadingTables(true);
      setError(null);
      
      // In a real implementation, this would call an API endpoint
      // For this demo, we'll use a mock response
      
      // Mock API call
      // const response = await apiClient.get(`/api/schemas/${schemaName}/tables`);
      // const data = response.data;
      
      // Mock data
      const mockTables: TableMetadata[] = [
        {
          table_name: 'TABLES',
          schema_name: schemaName,
          column_count: 8,
          row_count: 612,
          size_mb: 0.34,
          created: '2023-01-15',
          is_column_table: true,
        },
        {
          table_name: 'TABLE_COLUMNS',
          schema_name: schemaName,
          column_count: 12,
          row_count: 4823,
          size_mb: 1.67,
          created: '2023-01-15',
          is_column_table: true,
        },
      ];
      
      if (schemaName === 'CUSTOMER_DATA') {
        mockTables.push(
          {
            table_name: 'CUSTOMERS',
            schema_name: schemaName,
            column_count: 14,
            row_count: 25432,
            size_mb: 8.76,
            created: '2023-03-22',
            is_column_table: true,
          },
          {
            table_name: 'ADDRESSES',
            schema_name: schemaName,
            column_count: 10,
            row_count: 31254,
            size_mb: 6.21,
            created: '2023-03-22',
            is_column_table: true,
          },
          {
            table_name: 'CUSTOMER_PREFERENCES',
            schema_name: schemaName,
            column_count: 8,
            row_count: 18735,
            size_mb: 2.45,
            created: '2023-03-25',
            is_column_table: true,
          }
        );
      } else if (schemaName === 'SALES') {
        mockTables.push(
          {
            table_name: 'ORDERS',
            schema_name: schemaName,
            column_count: 12,
            row_count: 154783,
            size_mb: 24.56,
            created: '2023-04-10',
            is_column_table: true,
          },
          {
            table_name: 'ORDER_ITEMS',
            schema_name: schemaName,
            column_count: 8,
            row_count: 872451,
            size_mb: 45.78,
            created: '2023-04-10',
            is_column_table: true,
          },
          {
            table_name: 'INVOICES',
            schema_name: schemaName,
            column_count: 15,
            row_count: 154783,
            size_mb: 32.12,
            created: '2023-04-11',
            is_column_table: true,
          }
        );
      } else if (schemaName === 'PRODUCTS') {
        mockTables.push(
          {
            table_name: 'PRODUCT_CATALOG',
            schema_name: schemaName,
            column_count: 16,
            row_count: 12458,
            size_mb: 5.67,
            created: '2023-04-12',
            is_column_table: true,
          },
          {
            table_name: 'CATEGORIES',
            schema_name: schemaName,
            column_count: 6,
            row_count: 128,
            size_mb: 0.14,
            created: '2023-04-12',
            is_column_table: true,
          },
          {
            table_name: 'INVENTORY',
            schema_name: schemaName,
            column_count: 9,
            row_count: 12458,
            size_mb: 3.23,
            created: '2023-04-12',
            is_column_table: true,
          }
        );
      }
      
      setTables(prev => ({
        ...prev,
        [schemaName]: mockTables,
      }));
    } catch (err: any) {
      console.error(`Error fetching tables for schema ${schemaName}:`, err);
      setError(err.response?.data?.message || `Failed to fetch tables for schema ${schemaName}`);
    } finally {
      setIsLoadingTables(false);
    }
  };
  
  // Load columns for a table
  const fetchColumns = async (schemaName: string, tableName: string) => {
    try {
      setIsLoadingColumns(true);
      setError(null);
      
      // In a real implementation, this would call an API endpoint
      // For this demo, we'll use a mock response
      
      // Mock API call
      // const response = await apiClient.get(`/api/schemas/${schemaName}/tables/${tableName}/columns`);
      // const data = response.data;
      
      // Mock data
      let mockColumns: ColumnMetadata[] = [];
      
      if (tableName === 'CUSTOMERS' && schemaName === 'CUSTOMER_DATA') {
        mockColumns = [
          {
            column_name: 'CUSTOMER_ID',
            data_type: 'NVARCHAR',
            length: 36,
            nullable: false,
            position: 1,
            is_primary_key: true,
          },
          {
            column_name: 'FIRST_NAME',
            data_type: 'NVARCHAR',
            length: 100,
            nullable: false,
            position: 2,
            is_primary_key: false,
          },
          {
            column_name: 'LAST_NAME',
            data_type: 'NVARCHAR',
            length: 100,
            nullable: false,
            position: 3,
            is_primary_key: false,
          },
          {
            column_name: 'EMAIL',
            data_type: 'NVARCHAR',
            length: 255,
            nullable: false,
            position: 4,
            is_primary_key: false,
          },
          {
            column_name: 'PHONE',
            data_type: 'NVARCHAR',
            length: 20,
            nullable: true,
            position: 5,
            is_primary_key: false,
          },
          {
            column_name: 'BIRTH_DATE',
            data_type: 'DATE',
            length: 10,
            nullable: true,
            position: 6,
            is_primary_key: false,
          },
          {
            column_name: 'REGISTRATION_DATE',
            data_type: 'TIMESTAMP',
            length: 29,
            nullable: false,
            position: 7,
            is_primary_key: false,
          },
          {
            column_name: 'LAST_LOGIN',
            data_type: 'TIMESTAMP',
            length: 29,
            nullable: true,
            position: 8,
            is_primary_key: false,
          },
          {
            column_name: 'STATUS',
            data_type: 'NVARCHAR',
            length: 20,
            nullable: false,
            position: 9,
            is_primary_key: false,
          },
          {
            column_name: 'CUSTOMER_TYPE',
            data_type: 'NVARCHAR',
            length: 20,
            nullable: false,
            position: 10,
            is_primary_key: false,
          },
          {
            column_name: 'LOYALTY_POINTS',
            data_type: 'INTEGER',
            length: 10,
            nullable: false,
            position: 11,
            is_primary_key: false,
          },
          {
            column_name: 'NOTES',
            data_type: 'NCLOB',
            length: 1073741824,
            nullable: true,
            position: 12,
            is_primary_key: false,
          },
          {
            column_name: 'PREFERENCES_JSON',
            data_type: 'NCLOB',
            length: 1073741824,
            nullable: true,
            position: 13,
            is_primary_key: false,
          },
          {
            column_name: 'CREATED_BY',
            data_type: 'NVARCHAR',
            length: 100,
            nullable: false,
            position: 14,
            is_primary_key: false,
          },
        ];
      } else if (tableName === 'ORDERS' && schemaName === 'SALES') {
        mockColumns = [
          {
            column_name: 'ORDER_ID',
            data_type: 'NVARCHAR',
            length: 36,
            nullable: false,
            position: 1,
            is_primary_key: true,
          },
          {
            column_name: 'CUSTOMER_ID',
            data_type: 'NVARCHAR',
            length: 36,
            nullable: false,
            position: 2,
            is_primary_key: false,
          },
          {
            column_name: 'ORDER_DATE',
            data_type: 'TIMESTAMP',
            length: 29,
            nullable: false,
            position: 3,
            is_primary_key: false,
          },
          {
            column_name: 'STATUS',
            data_type: 'NVARCHAR',
            length: 20,
            nullable: false,
            position: 4,
            is_primary_key: false,
          },
          {
            column_name: 'TOTAL_AMOUNT',
            data_type: 'DECIMAL',
            length: 15,
            nullable: false,
            position: 5,
            is_primary_key: false,
          },
          {
            column_name: 'TAX_AMOUNT',
            data_type: 'DECIMAL',
            length: 15,
            nullable: false,
            position: 6,
            is_primary_key: false,
          },
          {
            column_name: 'SHIPPING_AMOUNT',
            data_type: 'DECIMAL',
            length: 15,
            nullable: false,
            position: 7,
            is_primary_key: false,
          },
          {
            column_name: 'SHIPPING_ADDRESS_ID',
            data_type: 'NVARCHAR',
            length: 36,
            nullable: false,
            position: 8,
            is_primary_key: false,
          },
          {
            column_name: 'BILLING_ADDRESS_ID',
            data_type: 'NVARCHAR',
            length: 36,
            nullable: false,
            position: 9,
            is_primary_key: false,
          },
          {
            column_name: 'PAYMENT_METHOD',
            data_type: 'NVARCHAR',
            length: 20,
            nullable: false,
            position: 10,
            is_primary_key: false,
          },
          {
            column_name: 'NOTES',
            data_type: 'NVARCHAR',
            length: 1000,
            nullable: true,
            position: 11,
            is_primary_key: false,
          },
          {
            column_name: 'CREATED_BY',
            data_type: 'NVARCHAR',
            length: 100,
            nullable: false,
            position: 12,
            is_primary_key: false,
          },
        ];
      } else if (tableName === 'PRODUCT_CATALOG' && schemaName === 'PRODUCTS') {
        mockColumns = [
          {
            column_name: 'PRODUCT_ID',
            data_type: 'NVARCHAR',
            length: 36,
            nullable: false,
            position: 1,
            is_primary_key: true,
          },
          {
            column_name: 'SKU',
            data_type: 'NVARCHAR',
            length: 50,
            nullable: false,
            position: 2,
            is_primary_key: false,
          },
          {
            column_name: 'NAME',
            data_type: 'NVARCHAR',
            length: 255,
            nullable: false,
            position: 3,
            is_primary_key: false,
          },
          {
            column_name: 'DESCRIPTION',
            data_type: 'NCLOB',
            length: 1073741824,
            nullable: true,
            position: 4,
            is_primary_key: false,
          },
          {
            column_name: 'CATEGORY_ID',
            data_type: 'NVARCHAR',
            length: 36,
            nullable: false,
            position: 5,
            is_primary_key: false,
          },
          {
            column_name: 'PRICE',
            data_type: 'DECIMAL',
            length: 15,
            nullable: false,
            position: 6,
            is_primary_key: false,
          },
          {
            column_name: 'COST',
            data_type: 'DECIMAL',
            length: 15,
            nullable: false,
            position: 7,
            is_primary_key: false,
          },
          {
            column_name: 'WEIGHT',
            data_type: 'DECIMAL',
            length: 10,
            nullable: true,
            position: 8,
            is_primary_key: false,
          },
          {
            column_name: 'DIMENSIONS',
            data_type: 'NVARCHAR',
            length: 100,
            nullable: true,
            position: 9,
            is_primary_key: false,
          },
          {
            column_name: 'RELEASE_DATE',
            data_type: 'DATE',
            length: 10,
            nullable: true,
            position: 10,
            is_primary_key: false,
          },
          {
            column_name: 'AVAILABLE',
            data_type: 'BOOLEAN',
            length: 1,
            nullable: false,
            position: 11,
            is_primary_key: false,
          },
          {
            column_name: 'FEATURED',
            data_type: 'BOOLEAN',
            length: 1,
            nullable: false,
            position: 12,
            is_primary_key: false,
          },
          {
            column_name: 'RATING',
            data_type: 'DECIMAL',
            length: 3,
            nullable: true,
            position: 13,
            is_primary_key: false,
          },
          {
            column_name: 'REVIEW_COUNT',
            data_type: 'INTEGER',
            length: 10,
            nullable: false,
            position: 14,
            is_primary_key: false,
          },
          {
            column_name: 'TAGS',
            data_type: 'NVARCHAR',
            length: 1000,
            nullable: true,
            position: 15,
            is_primary_key: false,
          },
          {
            column_name: 'CREATED_BY',
            data_type: 'NVARCHAR',
            length: 100,
            nullable: false,
            position: 16,
            is_primary_key: false,
          },
        ];
      } else {
        // Generic columns for other tables
        mockColumns = [
          {
            column_name: 'ID',
            data_type: 'NVARCHAR',
            length: 36,
            nullable: false,
            position: 1,
            is_primary_key: true,
          },
          {
            column_name: 'NAME',
            data_type: 'NVARCHAR',
            length: 100,
            nullable: false,
            position: 2,
            is_primary_key: false,
          },
          {
            column_name: 'DESCRIPTION',
            data_type: 'NVARCHAR',
            length: 500,
            nullable: true,
            position: 3,
            is_primary_key: false,
          },
          {
            column_name: 'CREATED_AT',
            data_type: 'TIMESTAMP',
            length: 29,
            nullable: false,
            position: 4,
            is_primary_key: false,
          },
          {
            column_name: 'UPDATED_AT',
            data_type: 'TIMESTAMP',
            length: 29,
            nullable: true,
            position: 5,
            is_primary_key: false,
          },
        ];
      }
      
      setColumns(mockColumns);
      
      // Load sample data
      fetchTableSample(schemaName, tableName);
    } catch (err: any) {
      console.error(`Error fetching columns for table ${schemaName}.${tableName}:`, err);
      setError(err.response?.data?.message || `Failed to fetch columns for table ${schemaName}.${tableName}`);
    } finally {
      setIsLoadingColumns(false);
    }
  };
  
  // Load sample data for a table
  const fetchTableSample = async (schemaName: string, tableName: string) => {
    try {
      setIsLoadingSample(true);
      setError(null);
      
      // In a real implementation, this would call an API endpoint
      // For this demo, we'll use a mock response
      
      // Mock API call
      // const response = await apiClient.get(`/api/schemas/${schemaName}/tables/${tableName}/sample`);
      // const data = response.data;
      
      // Mock data
      let mockSample: TableSample = {
        columns: [],
        rows: [],
      };
      
      if (tableName === 'CUSTOMERS' && schemaName === 'CUSTOMER_DATA') {
        mockSample = {
          columns: ['CUSTOMER_ID', 'FIRST_NAME', 'LAST_NAME', 'EMAIL', 'PHONE', 'STATUS'],
          rows: [
            ['C1001', 'John', 'Smith', 'john.smith@example.com', '+1-555-123-4567', 'ACTIVE'],
            ['C1002', 'Jane', 'Doe', 'jane.doe@example.com', '+1-555-987-6543', 'ACTIVE'],
            ['C1003', 'Robert', 'Johnson', 'robert.j@example.com', '+1-555-456-7890', 'INACTIVE'],
            ['C1004', 'Emily', 'Williams', 'emily.w@example.com', '+1-555-789-0123', 'ACTIVE'],
            ['C1005', 'Michael', 'Brown', 'michael.b@example.com', '+1-555-234-5678', 'ACTIVE'],
          ],
        };
      } else if (tableName === 'ORDERS' && schemaName === 'SALES') {
        mockSample = {
          columns: ['ORDER_ID', 'CUSTOMER_ID', 'ORDER_DATE', 'STATUS', 'TOTAL_AMOUNT'],
          rows: [
            ['O10001', 'C1001', '2023-06-15 10:23:45', 'COMPLETED', '129.99'],
            ['O10002', 'C1002', '2023-06-16 14:56:32', 'SHIPPED', '234.50'],
            ['O10003', 'C1001', '2023-06-18 09:34:21', 'PROCESSING', '56.75'],
            ['O10004', 'C1003', '2023-06-20 16:45:12', 'SHIPPED', '89.99'],
            ['O10005', 'C1005', '2023-06-21 11:22:33', 'COMPLETED', '176.25'],
          ],
        };
      } else if (tableName === 'PRODUCT_CATALOG' && schemaName === 'PRODUCTS') {
        mockSample = {
          columns: ['PRODUCT_ID', 'SKU', 'NAME', 'PRICE', 'CATEGORY_ID', 'AVAILABLE'],
          rows: [
            ['P1001', 'SKU-001', 'Ergonomic Office Chair', '249.99', 'CAT-001', 'TRUE'],
            ['P1002', 'SKU-002', 'Wireless Keyboard', '79.99', 'CAT-002', 'TRUE'],
            ['P1003', 'SKU-003', 'Ultra HD Monitor', '399.99', 'CAT-002', 'TRUE'],
            ['P1004', 'SKU-004', 'Wireless Mouse', '49.99', 'CAT-002', 'TRUE'],
            ['P1005', 'SKU-005', 'Laptop Stand', '39.99', 'CAT-003', 'FALSE'],
          ],
        };
      } else {
        // Generic sample for other tables
        mockSample = {
          columns: ['ID', 'NAME', 'DESCRIPTION', 'CREATED_AT'],
          rows: [
            ['1', 'Sample 1', 'Description for sample 1', '2023-01-15 10:00:00'],
            ['2', 'Sample 2', 'Description for sample 2', '2023-01-16 11:30:00'],
            ['3', 'Sample 3', 'Description for sample 3', '2023-01-17 12:45:00'],
            ['4', 'Sample 4', 'Description for sample 4', '2023-01-18 14:20:00'],
            ['5', 'Sample 5', 'Description for sample 5', '2023-01-19 16:05:00'],
          ],
        };
      }
      
      setTableSample(mockSample);
    } catch (err: any) {
      console.error(`Error fetching sample data for table ${schemaName}.${tableName}:`, err);
      setError(err.response?.data?.message || `Failed to fetch sample data for table ${schemaName}.${tableName}`);
    } finally {
      setIsLoadingSample(false);
    }
  };
  
  // Search for tables
  const searchTables = async () => {
    if (!searchTerm.trim()) return;
    
    try {
      setIsSearching(true);
      setError(null);
      
      // In a real implementation, this would call an API endpoint
      // For this demo, we'll use a mock response
      
      // Mock search results based on loaded tables
      const results: {schema: string, table: string}[] = [];
      
      Object.entries(tables).forEach(([schema, tableList]) => {
        tableList.forEach(table => {
          if (table.table_name.toLowerCase().includes(searchTerm.toLowerCase())) {
            results.push({
              schema: schema,
              table: table.table_name,
            });
          }
        });
      });
      
      // Add additional mock results if we don't have enough
      if (results.length === 0) {
        if (searchTerm.toLowerCase().includes('customer')) {
          results.push({
            schema: 'CUSTOMER_DATA',
            table: 'CUSTOMERS',
          });
        } else if (searchTerm.toLowerCase().includes('order')) {
          results.push({
            schema: 'SALES',
            table: 'ORDERS',
          });
        } else if (searchTerm.toLowerCase().includes('product')) {
          results.push({
            schema: 'PRODUCTS',
            table: 'PRODUCT_CATALOG',
          });
        }
      }
      
      setSearchResults(results);
    } catch (err: any) {
      console.error('Error searching tables:', err);
      setError(err.response?.data?.message || 'Failed to search tables');
    } finally {
      setIsSearching(false);
    }
  };
  
  // Toggle expanded schema
  const toggleSchema = (schemaName: string) => {
    if (expandedSchema === schemaName) {
      setExpandedSchema(null);
    } else {
      setExpandedSchema(schemaName);
      
      // Load tables if not already loaded
      if (!tables[schemaName]) {
        fetchTables(schemaName);
      }
    }
  };
  
  // Select a table
  const handleTableSelect = (schemaName: string, tableName: string) => {
    setSelectedTable({schema: schemaName, table: tableName});
    fetchColumns(schemaName, tableName);
    
    // Trigger callback if provided
    if (onTableSelect) {
      onTableSelect(schemaName, tableName);
    }
  };
  
  // Search on enter key
  const handleSearchKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      searchTables();
    }
  };
  
  // Initialize data on component mount
  useEffect(() => {
    fetchSchemas();
  }, []);
  
  return (
    <Card
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        borderRadius: { xs: 2, md: 3 },
        boxShadow: 3,
      }}
    >
      <CardContent sx={{ p: 0, flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <Box
          sx={{
            p: { xs: 2, sm: 3 },
            borderBottom: '1px solid',
            borderColor: 'divider',
          }}
        >
          <HumanText variant="h6" sx={{ fontWeight: 600, mb: 0.5 }}>
            SAP HANA Schema Explorer
          </HumanText>
          <HumanText variant="body2" color="text.secondary">
            Explore schemas, tables, and data in your SAP HANA Cloud database
          </HumanText>
        </Box>
        
        {/* Error message */}
        {error && (
          <Box sx={{ p: 2 }}>
            <Alert severity="error">{error}</Alert>
          </Box>
        )}
        
        {/* Main content */}
        <Box sx={{ flexGrow: 1, display: 'flex', overflow: 'hidden' }}>
          {/* Left panel - Schema/Table browser */}
          <Box
            sx={{
              width: '35%',
              borderRight: '1px solid',
              borderColor: 'divider',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            {/* Search box */}
            <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider' }}>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                  fullWidth
                  size="small"
                  placeholder="Search for tables..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  onKeyDown={handleSearchKeyDown}
                  InputProps={{
                    startAdornment: (
                      <SearchIcon color="action" sx={{ mr: 1, opacity: 0.5 }} />
                    ),
                  }}
                />
                <Button
                  variant="contained"
                  size="small"
                  onClick={searchTables}
                  disabled={isSearching || !searchTerm.trim()}
                  sx={{ minWidth: 'auto', px: 2 }}
                >
                  {isSearching ? <CircularProgress size={24} /> : 'Search'}
                </Button>
              </Box>
              
              {/* Search results */}
              {searchResults.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <HumanText variant="subtitle2" sx={{ mb: 1 }}>
                    Search Results:
                  </HumanText>
                  <List dense sx={{ bgcolor: alpha(theme.palette.primary.light, 0.1), borderRadius: 1 }}>
                    {searchResults.map((result, index) => (
                      <ListItem
                        button
                        key={index}
                        onClick={() => handleTableSelect(result.schema, result.table)}
                        selected={selectedTable?.schema === result.schema && selectedTable?.table === result.table}
                      >
                        <ListItemIcon sx={{ minWidth: 36 }}>
                          <StorageIcon fontSize="small" color="primary" />
                        </ListItemIcon>
                        <ListItemText
                          primary={result.table}
                          secondary={result.schema}
                          primaryTypographyProps={{ variant: 'body2' }}
                          secondaryTypographyProps={{ variant: 'caption' }}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}
            </Box>
            
            {/* Schema list */}
            <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
              {isLoadingSchemas ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                  <CircularProgress size={30} />
                </Box>
              ) : (
                <List sx={{ p: 0 }}>
                  {schemas.map((schema) => (
                    <React.Fragment key={schema.schema_name}>
                      <ListItem
                        button
                        onClick={() => toggleSchema(schema.schema_name)}
                        sx={{ px: 2 }}
                      >
                        <ListItemIcon>
                          {expandedSchema === schema.schema_name ? (
                            <FolderOpenIcon color="primary" />
                          ) : (
                            <FolderIcon color="primary" />
                          )}
                        </ListItemIcon>
                        <ListItemText
                          primary={schema.schema_name}
                          secondary={`${schema.table_count} tables, ${schema.view_count} views`}
                        />
                        {expandedSchema === schema.schema_name ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                      </ListItem>
                      
                      {/* Table list for the expanded schema */}
                      <Collapse in={expandedSchema === schema.schema_name} timeout="auto">
                        {isLoadingTables && !tables[schema.schema_name] ? (
                          <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
                            <CircularProgress size={24} />
                          </Box>
                        ) : (
                          <List component="div" disablePadding>
                            {tables[schema.schema_name]?.map((table) => (
                              <ListItem
                                button
                                key={table.table_name}
                                sx={{ pl: 6 }}
                                selected={selectedTable?.schema === schema.schema_name && selectedTable?.table === table.table_name}
                                onClick={() => handleTableSelect(schema.schema_name, table.table_name)}
                              >
                                <ListItemIcon sx={{ minWidth: 36 }}>
                                  <StorageIcon fontSize="small" />
                                </ListItemIcon>
                                <ListItemText
                                  primary={table.table_name}
                                  secondary={`${table.column_count} columns, ${table.row_count} rows`}
                                  primaryTypographyProps={{ variant: 'body2' }}
                                  secondaryTypographyProps={{ variant: 'caption' }}
                                />
                              </ListItem>
                            ))}
                          </List>
                        )}
                      </Collapse>
                    </React.Fragment>
                  ))}
                </List>
              )}
            </Box>
          </Box>
          
          {/* Right panel - Table details */}
          <Box
            sx={{
              width: '65%',
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}
          >
            {selectedTable ? (
              <>
                {/* Table header */}
                <Box
                  sx={{
                    p: 2,
                    borderBottom: '1px solid',
                    borderColor: 'divider',
                    bgcolor: alpha(theme.palette.primary.light, 0.05),
                  }}
                >
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Box>
                      <HumanText variant="subtitle1" sx={{ fontWeight: 600 }}>
                        {selectedTable.table}
                      </HumanText>
                      <HumanText variant="body2" color="text.secondary">
                        Schema: {selectedTable.schema}
                      </HumanText>
                    </Box>
                    <Button
                      variant="outlined"
                      size="small"
                      startIcon={<AutoAwesomeIcon />}
                      onClick={() => {
                        if (onTableSelect) {
                          onTableSelect(selectedTable.schema, selectedTable.table);
                        }
                      }}
                    >
                      Vectorize This Table
                    </Button>
                  </Box>
                </Box>
                
                {/* Table content */}
                <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
                  {isLoadingColumns ? (
                    <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                      <CircularProgress size={30} />
                    </Box>
                  ) : (
                    <>
                      {/* Column information */}
                      <HumanText variant="subtitle2" sx={{ mb: 1 }}>
                        Columns
                      </HumanText>
                      <TableContainer component={Paper} sx={{ mb: 3, maxHeight: 200, overflow: 'auto' }}>
                        <Table size="small" stickyHeader>
                          <TableHead>
                            <TableRow>
                              <TableCell>Name</TableCell>
                              <TableCell>Type</TableCell>
                              <TableCell>Length</TableCell>
                              <TableCell>Nullable</TableCell>
                              <TableCell>Primary Key</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {columns.map((column) => (
                              <TableRow key={column.column_name}>
                                <TableCell>{column.column_name}</TableCell>
                                <TableCell>{column.data_type}</TableCell>
                                <TableCell>{column.length}</TableCell>
                                <TableCell>{column.nullable ? 'Yes' : 'No'}</TableCell>
                                <TableCell>
                                  {column.is_primary_key && (
                                    <Chip
                                      label="PK"
                                      color="primary"
                                      size="small"
                                      sx={{ height: 20, fontSize: '0.7rem' }}
                                    />
                                  )}
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                      
                      {/* Sample data */}
                      <HumanText variant="subtitle2" sx={{ mb: 1 }}>
                        Sample Data
                      </HumanText>
                      {isLoadingSample ? (
                        <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
                          <CircularProgress size={24} />
                        </Box>
                      ) : tableSample ? (
                        <TableContainer component={Paper} sx={{ maxHeight: 300, overflow: 'auto' }}>
                          <Table size="small" stickyHeader>
                            <TableHead>
                              <TableRow>
                                {tableSample.columns.map((column) => (
                                  <TableCell key={column}>{column}</TableCell>
                                ))}
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {tableSample.rows.map((row, rowIndex) => (
                                <TableRow key={rowIndex}>
                                  {row.map((cell, cellIndex) => (
                                    <TableCell key={cellIndex}>{cell}</TableCell>
                                  ))}
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      ) : (
                        <Alert severity="info">No sample data available.</Alert>
                      )}
                      
                      {/* Data insights */}
                      <Box sx={{ mt: 3 }}>
                        <HumanText variant="subtitle2" sx={{ mb: 1 }}>
                          Data Insights
                        </HumanText>
                        <Grid container spacing={2}>
                          <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 2, bgcolor: alpha(theme.palette.success.light, 0.1) }}>
                              <HumanText variant="subtitle2" color="success.main" sx={{ mb: 1, display: 'flex', alignItems: 'center' }}>
                                <DataObjectIcon fontSize="small" sx={{ mr: 1 }} />
                                Column Characteristics
                              </HumanText>
                              <List dense>
                                <ListItem>
                                  <ListItemText
                                    primary="Text columns"
                                    secondary={`${columns.filter(c => c.data_type.includes('CHAR') || c.data_type.includes('CLOB')).length} columns`}
                                  />
                                </ListItem>
                                <ListItem>
                                  <ListItemText
                                    primary="Numeric columns"
                                    secondary={`${columns.filter(c => c.data_type.includes('INT') || c.data_type.includes('DEC') || c.data_type.includes('NUM')).length} columns`}
                                  />
                                </ListItem>
                                <ListItem>
                                  <ListItemText
                                    primary="Date/time columns"
                                    secondary={`${columns.filter(c => c.data_type.includes('DATE') || c.data_type.includes('TIME')).length} columns`}
                                  />
                                </ListItem>
                              </List>
                            </Paper>
                          </Grid>
                          <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 2, bgcolor: alpha(theme.palette.info.light, 0.1) }}>
                              <HumanText variant="subtitle2" color="info.main" sx={{ mb: 1, display: 'flex', alignItems: 'center' }}>
                                <InfoIcon fontSize="small" sx={{ mr: 1 }} />
                                Vectorization Potential
                              </HumanText>
                              <HumanText variant="body2" paragraph>
                                This table contains {columns.filter(c => c.data_type.includes('CHAR') || c.data_type.includes('CLOB')).length} text columns that can be vectorized for semantic search.
                              </HumanText>
                              <Button
                                fullWidth
                                variant="outlined"
                                color="primary"
                                startIcon={<AutoAwesomeIcon />}
                                onClick={() => {
                                  if (onTableSelect) {
                                    onTableSelect(selectedTable.schema, selectedTable.table);
                                  }
                                }}
                              >
                                Start Vectorization Process
                              </Button>
                            </Paper>
                          </Grid>
                        </Grid>
                      </Box>
                    </>
                  )}
                </Box>
              </>
            ) : (
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  height: '100%',
                  flexDirection: 'column',
                  p: 3,
                }}
              >
                <StorageIcon sx={{ fontSize: 60, color: alpha(theme.palette.primary.main, 0.2), mb: 2 }} />
                <HumanText variant="h6" sx={{ mb: 1, textAlign: 'center' }}>
                  Select a Table
                </HumanText>
                <HumanText
                  variant="body2"
                  color="text.secondary"
                  sx={{ maxWidth: 400, textAlign: 'center' }}
                >
                  Select a table from the schema browser to view its details and start the vectorization process.
                </HumanText>
              </Box>
            )}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default SchemaExplorer;