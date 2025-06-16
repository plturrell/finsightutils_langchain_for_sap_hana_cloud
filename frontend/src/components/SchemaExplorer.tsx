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
import { useSpring, animated, config, useTrail, useChain, useSpringRef } from '@react-spring/web';
import HumanText from './HumanText';
import apiClient from '../api/client';

// Create animated versions of MUI components
const AnimatedBox = animated(Box);
const AnimatedCard = animated(Card);
const AnimatedPaper = animated(Paper);
const AnimatedTypography = animated(Typography);
const AnimatedButton = animated(Button);
const AnimatedTextField = animated(TextField);
const AnimatedListItem = animated(ListItem);
const AnimatedTableContainer = animated(TableContainer);
const AnimatedAlert = animated(Alert);
const AnimatedChip = animated(Chip);

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
  
  // Animation states
  const [animationsVisible, setAnimationsVisible] = useState<boolean>(false);
  
  // Animation refs for chained animations
  const headerSpringRef = useSpringRef();
  const searchSpringRef = useSpringRef();
  const schemasSpringRef = useSpringRef();
  const tablesSpringRef = useSpringRef();
  const detailsSpringRef = useSpringRef();
  const columnsSpringRef = useSpringRef();
  const sampleSpringRef = useSpringRef();
  const insightsSpringRef = useSpringRef();
  
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
    
    // Trigger animations after a short delay
    const timer = setTimeout(() => {
      setAnimationsVisible(true);
    }, 200);
    
    return () => clearTimeout(timer);
  }, []);
  
  // Header animation
  const headerAnimation = useSpring({
    ref: headerSpringRef,
    from: { opacity: 0, transform: 'translateY(-20px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(-20px)' },
    config: { tension: 280, friction: 60 }
  });
  
  // Search box animation
  const searchAnimation = useSpring({
    ref: searchSpringRef,
    from: { opacity: 0, transform: 'translateY(-15px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(-15px)' },
    config: { tension: 280, friction: 60 }
  });
  
  // Schema list animation trail
  const schemaTrail = useTrail(schemas.length, {
    ref: schemasSpringRef,
    from: { opacity: 0, transform: 'translateX(-20px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateX(0)' : 'translateX(-20px)' },
    config: { mass: 1, tension: 280, friction: 60 }
  });
  
  // Selected table details animation
  const detailsAnimation = useSpring({
    ref: detailsSpringRef,
    from: { opacity: 0, transform: 'scale(0.95)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'scale(1)' : 'scale(0.95)' },
    config: { tension: 280, friction: 60 }
  });
  
  // Columns table animation
  const columnsAnimation = useSpring({
    ref: columnsSpringRef,
    from: { opacity: 0, transform: 'translateY(20px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(20px)' },
    config: { tension: 280, friction: 60 }
  });
  
  // Sample data animation
  const sampleAnimation = useSpring({
    ref: sampleSpringRef,
    from: { opacity: 0, transform: 'translateY(20px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(20px)' },
    config: { tension: 280, friction: 60 }
  });
  
  // Insights animation
  const insightsAnimation = useSpring({
    ref: insightsSpringRef,
    from: { opacity: 0, transform: 'translateY(20px)' },
    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(20px)' },
    config: { tension: 280, friction: 60 }
  });
  
  // Search results animation
  const searchResultsAnimation = useSpring({
    from: { opacity: 0, height: 0, transform: 'translateY(-10px)' },
    to: { 
      opacity: searchResults.length > 0 ? 1 : 0, 
      height: searchResults.length > 0 ? 'auto' : 0,
      transform: searchResults.length > 0 ? 'translateY(0)' : 'translateY(-10px)'
    },
    config: { tension: 280, friction: 60 }
  });
  
  // No selection state animation
  const noSelectionAnimation = useSpring({
    from: { opacity: 0, transform: 'scale(0.9)' },
    to: { 
      opacity: selectedTable ? 0 : (animationsVisible ? 1 : 0), 
      transform: selectedTable ? 'scale(0.85)' : (animationsVisible ? 'scale(1)' : 'scale(0.9)')
    },
    config: { tension: 280, friction: 60 }
  });
  
  // Chain the animations in sequence
  useChain(
    animationsVisible
      ? [headerSpringRef, searchSpringRef, schemasSpringRef, detailsSpringRef, columnsSpringRef, sampleSpringRef, insightsSpringRef]
      : [insightsSpringRef, sampleSpringRef, columnsSpringRef, detailsSpringRef, schemasSpringRef, searchSpringRef, headerSpringRef],
    animationsVisible
      ? [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
      : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
  );
  
  return (
    <AnimatedCard
      style={useSpring({
        from: { opacity: 0, transform: 'translateY(30px)' },
        to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(30px)' },
        config: { tension: 280, friction: 60 }
      })}
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        borderRadius: { xs: 2, md: 3 },
        boxShadow: '0 6px 20px rgba(0, 0, 0, 0.05)',
        border: '1px solid rgba(0, 102, 179, 0.1)',
        transition: 'box-shadow 0.3s ease',
        '&:hover': {
          boxShadow: '0 10px 25px rgba(0, 0, 0, 0.08)',
        },
        overflow: 'hidden',
      }}
    >
      <CardContent sx={{ p: 0, flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <AnimatedBox
          style={headerAnimation}
          sx={{
            p: { xs: 2, sm: 3 },
            borderBottom: '1px solid',
            borderColor: 'divider',
            background: 'linear-gradient(180deg, rgba(0, 102, 179, 0.02) 0%, rgba(255, 255, 255, 0) 100%)',
          }}
        >
          <AnimatedTypography 
            component={HumanText}
            variant="h6" 
            style={{
              background: 'linear-gradient(90deg, #0066B3 0%, #19B5FE 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundSize: '200% 100%',
              backgroundPosition: 'right bottom',
              ...useSpring({
                from: { backgroundPosition: '0% 50%' },
                to: { backgroundPosition: '100% 50%' },
                config: { duration: 3000 },
                loop: { reverse: true }
              })
            }}
            sx={{ fontWeight: 600, mb: 0.5 }}
          >
            SAP HANA Schema Explorer
          </AnimatedTypography>
          <AnimatedTypography 
            component={HumanText}
            variant="body2"
            style={useSpring({
              from: { opacity: 0, transform: 'translateY(5px)' },
              to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(5px)' },
              delay: 100,
              config: { tension: 280, friction: 60 }
            })}
            color="text.secondary"
          >
            Explore schemas, tables, and data in your SAP HANA Cloud database
          </AnimatedTypography>
        </AnimatedBox>
        
        {/* Error message */}
        {error && (
          <AnimatedBox 
            style={useSpring({
              from: { opacity: 0, transform: 'translateY(-10px)' },
              to: { opacity: 1, transform: 'translateY(0)' },
              config: { tension: 280, friction: 60 }
            })}
            sx={{ p: 2 }}
          >
            <AnimatedAlert 
              severity="error"
              style={useSpring({
                from: { opacity: 0, transform: 'scale(0.95)' },
                to: { opacity: 1, transform: 'scale(1)' },
                config: { tension: 280, friction: 60 }
              })}
            >
              {error}
            </AnimatedAlert>
          </AnimatedBox>
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
            <AnimatedBox 
              style={searchAnimation}
              sx={{ 
                p: 2, 
                borderBottom: '1px solid', 
                borderColor: 'divider',
                background: 'linear-gradient(180deg, rgba(255, 255, 255, 0.8) 0%, rgba(255, 255, 255, 0.4) 100%)',
              }}
            >
              <AnimatedBox 
                style={useSpring({
                  from: { opacity: 0, transform: 'translateY(10px)' },
                  to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(10px)' },
                  delay: 150,
                  config: { tension: 280, friction: 60 }
                })}
                sx={{ display: 'flex', gap: 1 }}
              >
                <AnimatedTextField
                  fullWidth
                  size="small"
                  placeholder="Search for tables..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  onKeyDown={handleSearchKeyDown}
                  style={useSpring({
                    from: { opacity: 0, transform: 'translateX(-10px)' },
                    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateX(0)' : 'translateX(-10px)' },
                    delay: 200,
                    config: { tension: 280, friction: 60 }
                  })}
                  InputProps={{
                    startAdornment: (
                      <SearchIcon color="action" sx={{ 
                        mr: 1, 
                        opacity: 0.5,
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          opacity: 0.8,
                          transform: 'scale(1.1)',
                        }
                      }} />
                    ),
                    sx: {
                      borderRadius: 2,
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        boxShadow: '0 2px 8px rgba(0, 102, 179, 0.1)',
                      },
                      '&:focus-within': {
                        boxShadow: '0 2px 10px rgba(0, 102, 179, 0.15)',
                      }
                    }
                  }}
                />
                <AnimatedButton
                  variant="contained"
                  size="small"
                  onClick={searchTables}
                  disabled={isSearching || !searchTerm.trim()}
                  style={useSpring({
                    from: { opacity: 0, transform: 'translateX(10px)' },
                    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateX(0)' : 'translateX(10px)' },
                    delay: 250,
                    config: { tension: 280, friction: 60 }
                  })}
                  sx={{ 
                    minWidth: 'auto', 
                    px: 2,
                    borderRadius: 2,
                    background: 'linear-gradient(90deg, #0066B3, #19B5FE)',
                    boxShadow: '0 4px 10px rgba(0, 102, 179, 0.2)',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: '0 6px 14px rgba(0, 102, 179, 0.3)',
                    },
                    '&:active': {
                      transform: 'translateY(0px)',
                    }
                  }}
                >
                  {isSearching ? <CircularProgress size={24} color="inherit" /> : 'Search'}
                </AnimatedButton>
              </AnimatedBox>
              
              {/* Search results */}
              <AnimatedBox style={searchResultsAnimation}>
                {searchResults.length > 0 && (
                  <Box sx={{ mt: 2 }}>
                    <AnimatedTypography 
                      component={HumanText} 
                      variant="subtitle2" 
                      sx={{ mb: 1 }}
                      style={useSpring({
                        from: { opacity: 0, transform: 'translateY(5px)' },
                        to: { opacity: 1, transform: 'translateY(0)' },
                        config: { tension: 280, friction: 60 }
                      })}
                    >
                      Search Results:
                    </AnimatedTypography>
                    <List dense sx={{ 
                      bgcolor: alpha(theme.palette.primary.light, 0.1), 
                      borderRadius: 2,
                      overflow: 'hidden',
                      border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
                      boxShadow: '0 4px 12px rgba(0, 102, 179, 0.05)'
                    }}>
                      {searchResults.map((result, index) => {
                        const isSelected = selectedTable?.schema === result.schema && selectedTable?.table === result.table;
                        
                        return (
                          <AnimatedListItem
                            button
                            key={index}
                            onClick={() => handleTableSelect(result.schema, result.table)}
                            selected={isSelected}
                            style={useSpring({
                              from: { opacity: 0, transform: 'translateY(10px)' },
                              to: { opacity: 1, transform: 'translateY(0)' },
                              delay: 50 * index,
                              config: { tension: 280, friction: 60 }
                            })}
                            sx={{
                              transition: 'all 0.3s ease',
                              '&.Mui-selected': {
                                backgroundColor: alpha(theme.palette.primary.main, 0.1),
                                '&:hover': {
                                  backgroundColor: alpha(theme.palette.primary.main, 0.15),
                                }
                              },
                              '&:hover': {
                                backgroundColor: alpha(theme.palette.primary.main, 0.05),
                                transform: 'translateX(3px)'
                              }
                            }}
                          >
                            <ListItemIcon sx={{ 
                              minWidth: 36,
                              transition: 'transform 0.3s ease',
                              transform: isSelected ? 'scale(1.2)' : 'scale(1)',
                            }}>
                              <StorageIcon 
                                fontSize="small" 
                                color={isSelected ? "primary" : "action"}
                                sx={{
                                  transition: 'all 0.3s ease',
                                  filter: isSelected ? 'drop-shadow(0 2px 3px rgba(0, 102, 179, 0.3))' : 'none'
                                }}
                              />
                            </ListItemIcon>
                            <ListItemText
                              primary={<HumanText>{result.table}</HumanText>}
                              secondary={<HumanText>{result.schema}</HumanText>}
                              primaryTypographyProps={{ 
                                variant: 'body2',
                                sx: {
                                  transition: 'all 0.3s ease',
                                  fontWeight: isSelected ? 600 : 400,
                                  color: isSelected ? theme.palette.primary.main : theme.palette.text.primary,
                                }
                              }}
                              secondaryTypographyProps={{ variant: 'caption' }}
                            />
                          </AnimatedListItem>
                        );
                      })}
                    </List>
                  </Box>
                )}
              </AnimatedBox>
            </Box>
            
            {/* Schema list */}
            <AnimatedBox 
              style={useSpring({
                from: { opacity: 0 },
                to: { opacity: animationsVisible ? 1 : 0 },
                delay: 300,
                config: { tension: 280, friction: 60 }
              })}
              sx={{ flexGrow: 1, overflow: 'auto' }}
            >
              {isLoadingSchemas ? (
                <AnimatedBox 
                  style={useSpring({
                    from: { opacity: 0 },
                    to: { opacity: 1 },
                    config: { tension: 280, friction: 60 }
                  })}
                  sx={{ display: 'flex', justifyContent: 'center', p: 3 }}
                >
                  <CircularProgress 
                    size={30} 
                    sx={{
                      color: theme.palette.primary.main,
                      animation: 'pulse 1.5s ease-in-out infinite',
                      '@keyframes pulse': {
                        '0%': {
                          opacity: 0.6,
                          transform: 'scale(0.95)',
                        },
                        '50%': {
                          opacity: 1,
                          transform: 'scale(1.05)',
                        },
                        '100%': {
                          opacity: 0.6,
                          transform: 'scale(0.95)',
                        },
                      },
                    }}
                  />
                </AnimatedBox>
              ) : (
                <List sx={{ p: 0 }}>
                  {schemaTrail.map((style, index) => {
                    const schema = schemas[index];
                    const isExpanded = expandedSchema === schema.schema_name;
                    
                    return (
                      <React.Fragment key={schema.schema_name}>
                        <AnimatedListItem
                          button
                          onClick={() => toggleSchema(schema.schema_name)}
                          style={style}
                          sx={{ 
                            px: 2,
                            transition: 'all 0.3s ease',
                            '&:hover': {
                              backgroundColor: alpha(theme.palette.primary.main, 0.05),
                              transform: 'translateX(3px)'
                            }
                          }}
                        >
                          <ListItemIcon sx={{
                            transition: 'transform 0.3s ease',
                            transform: isExpanded ? 'scale(1.1)' : 'scale(1)',
                          }}>
                            {isExpanded ? (
                              <FolderOpenIcon 
                                color="primary" 
                                sx={{
                                  transition: 'all 0.3s ease',
                                  filter: 'drop-shadow(0 2px 3px rgba(0, 102, 179, 0.3))'
                                }}
                              />
                            ) : (
                              <FolderIcon 
                                color="primary"
                                sx={{
                                  transition: 'all 0.3s ease',
                                  '&:hover': {
                                    filter: 'drop-shadow(0 1px 2px rgba(0, 102, 179, 0.2))'
                                  }
                                }}
                              />
                            )}
                          </ListItemIcon>
                          <ListItemText
                            primary={<HumanText>{schema.schema_name}</HumanText>}
                            secondary={<HumanText>{`${schema.table_count} tables, ${schema.view_count} views`}</HumanText>}
                            primaryTypographyProps={{
                              sx: {
                                fontWeight: isExpanded ? 600 : 400,
                                color: isExpanded ? theme.palette.primary.main : theme.palette.text.primary,
                                transition: 'all 0.3s ease',
                                ...(isExpanded && {
                                  background: 'linear-gradient(90deg, #0066B3 0%, #19B5FE 100%)',
                                  WebkitBackgroundClip: 'text',
                                  WebkitTextFillColor: 'transparent',
                                })
                              }
                            }}
                          />
                          <animated.div style={useSpring({
                            transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                            config: { tension: 200, friction: 20 }
                          })}>
                            <ExpandMoreIcon sx={{ 
                              color: isExpanded ? theme.palette.primary.main : theme.palette.text.secondary,
                              transition: 'color 0.3s ease'
                            }} />
                          </animated.div>
                        </AnimatedListItem>
                        
                        {/* Table list for the expanded schema */}
                        <Collapse in={isExpanded} timeout="auto">
                          {isLoadingTables && !tables[schema.schema_name] ? (
                            <AnimatedBox 
                              style={useSpring({
                                from: { opacity: 0 },
                                to: { opacity: 1 },
                                config: { tension: 280, friction: 60 }
                              })}
                              sx={{ display: 'flex', justifyContent: 'center', p: 2 }}
                            >
                              <CircularProgress size={24} />
                            </AnimatedBox>
                          ) : (
                            <List component="div" disablePadding>
                              {tables[schema.schema_name]?.map((table, tableIndex) => {
                                const isSelected = selectedTable?.schema === schema.schema_name && selectedTable?.table === table.table_name;
                                
                                return (
                                  <AnimatedListItem
                                    button
                                    key={table.table_name}
                                    style={useSpring({
                                      from: { opacity: 0, transform: 'translateX(-20px)' },
                                      to: { opacity: isExpanded ? 1 : 0, transform: isExpanded ? 'translateX(0)' : 'translateX(-20px)' },
                                      delay: 30 * tableIndex,
                                      config: { tension: 280, friction: 60 }
                                    })}
                                    sx={{ 
                                      pl: 6,
                                      transition: 'all 0.3s ease',
                                      '&.Mui-selected': {
                                        backgroundColor: alpha(theme.palette.primary.main, 0.1),
                                        '&:hover': {
                                          backgroundColor: alpha(theme.palette.primary.main, 0.15),
                                        }
                                      },
                                      '&:hover': {
                                        backgroundColor: alpha(theme.palette.primary.main, 0.05),
                                        transform: 'translateX(5px)'
                                      }
                                    }}
                                    selected={isSelected}
                                    onClick={() => handleTableSelect(schema.schema_name, table.table_name)}
                                  >
                                    <ListItemIcon sx={{ 
                                      minWidth: 36,
                                      transition: 'transform 0.3s ease',
                                      transform: isSelected ? 'scale(1.2)' : 'scale(1)',
                                    }}>
                                      <StorageIcon 
                                        fontSize="small"
                                        color={isSelected ? "primary" : "action"}
                                        sx={{
                                          transition: 'all 0.3s ease',
                                          filter: isSelected ? 'drop-shadow(0 2px 3px rgba(0, 102, 179, 0.3))' : 'none'
                                        }}
                                      />
                                    </ListItemIcon>
                                    <ListItemText
                                      primary={<HumanText>{table.table_name}</HumanText>}
                                      secondary={<HumanText>{`${table.column_count} columns, ${table.row_count} rows`}</HumanText>}
                                      primaryTypographyProps={{ 
                                        variant: 'body2',
                                        sx: {
                                          transition: 'all 0.3s ease',
                                          fontWeight: isSelected ? 600 : 400,
                                          color: isSelected ? theme.palette.primary.main : theme.palette.text.primary,
                                          ...(isSelected && {
                                            textShadow: '0 0 1px rgba(0, 102, 179, 0.2)'
                                          })
                                        }
                                      }}
                                      secondaryTypographyProps={{ variant: 'caption' }}
                                    />
                                  </AnimatedListItem>
                                );
                              })}
                            </List>
                          )}
                        </Collapse>
                      </React.Fragment>
                    );
                  })}
                </List>
              )}
            </AnimatedBox>
          </Box>
          
          {/* Right panel - Table details */}
          <AnimatedBox
            style={useSpring({
              from: { opacity: 0 },
              to: { opacity: animationsVisible ? 1 : 0 },
              delay: 400,
              config: { tension: 280, friction: 60 }
            })}
            sx={{
              width: '65%',
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
              borderLeft: '1px solid',
              borderColor: 'divider',
            }}
          >
            {selectedTable ? (
              <>
                {/* Table header */}
                <AnimatedBox
                  style={detailsAnimation}
                  sx={{
                    p: 2,
                    borderBottom: '1px solid',
                    borderColor: 'divider',
                    background: 'linear-gradient(90deg, rgba(0, 102, 179, 0.05) 0%, rgba(255, 255, 255, 0) 100%)',
                  }}
                >
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <AnimatedBox 
                      style={useSpring({
                        from: { opacity: 0, transform: 'translateX(-10px)' },
                        to: { opacity: 1, transform: 'translateX(0)' },
                        config: { tension: 280, friction: 60 }
                      })}
                    >
                      <AnimatedTypography 
                        component={HumanText} 
                        variant="subtitle1" 
                        style={{
                          background: 'linear-gradient(90deg, #0066B3 0%, #19B5FE 100%)',
                          WebkitBackgroundClip: 'text',
                          WebkitTextFillColor: 'transparent',
                        }}
                        sx={{ fontWeight: 600 }}
                      >
                        {selectedTable.table}
                      </AnimatedTypography>
                      <AnimatedTypography 
                        component={HumanText}
                        variant="body2" 
                        color="text.secondary"
                        style={useSpring({
                          from: { opacity: 0, transform: 'translateY(5px)' },
                          to: { opacity: 1, transform: 'translateY(0)' },
                          delay: 100,
                          config: { tension: 280, friction: 60 }
                        })}
                      >
                        Schema: {selectedTable.schema}
                      </AnimatedTypography>
                    </AnimatedBox>
                    <AnimatedButton
                      variant="outlined"
                      size="small"
                      startIcon={<AutoAwesomeIcon />}
                      onClick={() => {
                        if (onTableSelect) {
                          onTableSelect(selectedTable.schema, selectedTable.table);
                        }
                      }}
                      style={useSpring({
                        from: { opacity: 0, transform: 'translateX(10px)' },
                        to: { opacity: 1, transform: 'translateX(0)' },
                        delay: 200,
                        config: { tension: 280, friction: 60 }
                      })}
                      sx={{ 
                        borderRadius: 2,
                        borderColor: theme.palette.primary.main,
                        color: theme.palette.primary.main,
                        position: 'relative',
                        overflow: 'hidden',
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          transform: 'translateY(-2px)',
                          boxShadow: '0 4px 8px rgba(0, 102, 179, 0.2)',
                          borderColor: theme.palette.primary.main,
                          backgroundColor: alpha(theme.palette.primary.main, 0.04),
                          '&::after': {
                            opacity: 1,
                            transform: 'translateX(100%)',
                          }
                        },
                        '&::after': {
                          content: '""',
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          width: '100%',
                          height: '100%',
                          background: `linear-gradient(90deg, transparent, ${alpha(theme.palette.primary.main, 0.2)}, transparent)`,
                          opacity: 0,
                          transform: 'translateX(-100%)',
                          transition: 'transform 0.6s ease, opacity 0.6s ease',
                        }
                      }}
                    >
                      Vectorize This Table
                    </AnimatedButton>
                  </Box>
                </AnimatedBox>
                
                {/* Table content */}
                <AnimatedBox 
                  style={useSpring({
                    from: { opacity: 0 },
                    to: { opacity: 1 },
                    delay: 300,
                    config: { tension: 280, friction: 60 }
                  })}
                  sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}
                >
                  {isLoadingColumns ? (
                    <AnimatedBox 
                      style={useSpring({
                        from: { opacity: 0 },
                        to: { opacity: 1 },
                        config: { tension: 280, friction: 60 }
                      })}
                      sx={{ display: 'flex', justifyContent: 'center', p: 3 }}
                    >
                      <CircularProgress 
                        size={30} 
                        sx={{
                          color: theme.palette.primary.main,
                          animation: 'pulse 1.5s ease-in-out infinite',
                          '@keyframes pulse': {
                            '0%': {
                              opacity: 0.6,
                              transform: 'scale(0.95)',
                            },
                            '50%': {
                              opacity: 1,
                              transform: 'scale(1.05)',
                            },
                            '100%': {
                              opacity: 0.6,
                              transform: 'scale(0.95)',
                            },
                          },
                        }}
                      />
                    </AnimatedBox>
                  ) : (
                    <>
                      {/* Column information */}
                      <AnimatedTypography 
                        component={HumanText}
                        variant="subtitle2" 
                        sx={{ mb: 1 }}
                        style={useSpring({
                          from: { opacity: 0, transform: 'translateY(10px)' },
                          to: { opacity: 1, transform: 'translateY(0)' },
                          config: { tension: 280, friction: 60 }
                        })}
                      >
                        Columns
                      </AnimatedTypography>
                      <AnimatedTableContainer 
                        component={AnimatedPaper} 
                        style={columnsAnimation}
                        sx={{ 
                          mb: 3, 
                          maxHeight: 200, 
                          overflow: 'auto',
                          borderRadius: 2,
                          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)',
                          border: '1px solid rgba(0, 102, 179, 0.1)',
                        }}
                      >
                        <Table size="small" stickyHeader>
                          <TableHead>
                            <TableRow>
                              <TableCell sx={{ 
                                fontWeight: 600,
                                background: alpha(theme.palette.primary.main, 0.05),
                              }}>Name</TableCell>
                              <TableCell sx={{ 
                                fontWeight: 600,
                                background: alpha(theme.palette.primary.main, 0.05),
                              }}>Type</TableCell>
                              <TableCell sx={{ 
                                fontWeight: 600,
                                background: alpha(theme.palette.primary.main, 0.05),
                              }}>Length</TableCell>
                              <TableCell sx={{ 
                                fontWeight: 600,
                                background: alpha(theme.palette.primary.main, 0.05),
                              }}>Nullable</TableCell>
                              <TableCell sx={{ 
                                fontWeight: 600,
                                background: alpha(theme.palette.primary.main, 0.05),
                              }}>Primary Key</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {columns.map((column, index) => (
                              <TableRow 
                                key={column.column_name}
                                sx={{ 
                                  '&:nth-of-type(odd)': {
                                    backgroundColor: alpha(theme.palette.action.hover, 0.05),
                                  },
                                  transition: 'background-color 0.3s ease',
                                  '&:hover': {
                                    backgroundColor: alpha(theme.palette.primary.main, 0.03),
                                  },
                                  // Animation for row entrance
                                  animation: `fadeIn 0.5s ease forwards ${index * 0.05}s`,
                                  opacity: 0,
                                  '@keyframes fadeIn': {
                                    from: { opacity: 0, transform: 'translateY(10px)' },
                                    to: { opacity: 1, transform: 'translateY(0)' }
                                  }
                                }}
                              >
                                <TableCell sx={{ 
                                  fontWeight: column.is_primary_key ? 600 : 400,
                                  color: column.is_primary_key ? theme.palette.primary.main : 'inherit',
                                }}>
                                  {column.column_name}
                                </TableCell>
                                <TableCell>{column.data_type}</TableCell>
                                <TableCell>{column.length}</TableCell>
                                <TableCell>{column.nullable ? 'Yes' : 'No'}</TableCell>
                                <TableCell>
                                  {column.is_primary_key && (
                                    <AnimatedChip
                                      label="PK"
                                      color="primary"
                                      size="small"
                                      style={useSpring({
                                        from: { opacity: 0, transform: 'scale(0.8)' },
                                        to: { opacity: 1, transform: 'scale(1)' },
                                        delay: 100 + (index * 30),
                                        config: { tension: 280, friction: 20 }
                                      })}
                                      sx={{ 
                                        height: 20, 
                                        fontSize: '0.7rem',
                                        boxShadow: '0 2px 4px rgba(0, 102, 179, 0.2)',
                                      }}
                                    />
                                  )}
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </AnimatedTableContainer>
                      
                      {/* Sample data */}
                      <AnimatedTypography 
                        component={HumanText}
                        variant="subtitle2" 
                        sx={{ mb: 1 }}
                        style={useSpring({
                          from: { opacity: 0, transform: 'translateY(10px)' },
                          to: { opacity: 1, transform: 'translateY(0)' },
                          delay: 150,
                          config: { tension: 280, friction: 60 }
                        })}
                      >
                        Sample Data
                      </AnimatedTypography>
                      {isLoadingSample ? (
                        <AnimatedBox 
                          style={useSpring({
                            from: { opacity: 0 },
                            to: { opacity: 1 },
                            config: { tension: 280, friction: 60 }
                          })}
                          sx={{ display: 'flex', justifyContent: 'center', p: 2 }}
                        >
                          <CircularProgress size={24} />
                        </AnimatedBox>
                      ) : tableSample ? (
                        <AnimatedTableContainer 
                          component={AnimatedPaper} 
                          style={sampleAnimation}
                          sx={{ 
                            maxHeight: 300, 
                            overflow: 'auto',
                            borderRadius: 2,
                            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)',
                            border: '1px solid rgba(0, 102, 179, 0.1)',
                          }}
                        >
                          <Table size="small" stickyHeader>
                            <TableHead>
                              <TableRow>
                                {tableSample.columns.map((column) => (
                                  <TableCell 
                                    key={column}
                                    sx={{ 
                                      fontWeight: 600,
                                      background: alpha(theme.palette.primary.main, 0.05),
                                    }}
                                  >
                                    {column}
                                  </TableCell>
                                ))}
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {tableSample.rows.map((row, rowIndex) => (
                                <TableRow 
                                  key={rowIndex}
                                  sx={{ 
                                    '&:nth-of-type(odd)': {
                                      backgroundColor: alpha(theme.palette.action.hover, 0.05),
                                    },
                                    transition: 'background-color 0.3s ease',
                                    '&:hover': {
                                      backgroundColor: alpha(theme.palette.primary.main, 0.03),
                                    },
                                    // Animation for row entrance
                                    animation: `fadeIn 0.5s ease forwards ${rowIndex * 0.05}s`,
                                    opacity: 0,
                                    '@keyframes fadeIn': {
                                      from: { opacity: 0, transform: 'translateY(10px)' },
                                      to: { opacity: 1, transform: 'translateY(0)' }
                                    }
                                  }}
                                >
                                  {row.map((cell, cellIndex) => (
                                    <TableCell key={cellIndex}>{cell}</TableCell>
                                  ))}
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </AnimatedTableContainer>
                      ) : (
                        <AnimatedAlert 
                          severity="info"
                          style={useSpring({
                            from: { opacity: 0, transform: 'translateY(10px)' },
                            to: { opacity: 1, transform: 'translateY(0)' },
                            config: { tension: 280, friction: 60 }
                          })}
                        >
                          No sample data available.
                        </AnimatedAlert>
                      )}
                      
                      {/* Data insights */}
                      <AnimatedBox 
                        style={insightsAnimation}
                        sx={{ mt: 3 }}
                      >
                        <AnimatedTypography 
                          component={HumanText}
                          variant="subtitle2" 
                          sx={{ mb: 1 }}
                          style={useSpring({
                            from: { opacity: 0, transform: 'translateY(10px)' },
                            to: { opacity: 1, transform: 'translateY(0)' },
                            delay: 200,
                            config: { tension: 280, friction: 60 }
                          })}
                        >
                          Data Insights
                        </AnimatedTypography>
                        <Grid container spacing={2}>
                          <Grid item xs={12} md={6}>
                            <AnimatedPaper 
                              style={useSpring({
                                from: { opacity: 0, transform: 'translateY(20px)' },
                                to: { opacity: 1, transform: 'translateY(0)' },
                                delay: 250,
                                config: { tension: 280, friction: 60 }
                              })}
                              sx={{ 
                                p: 2, 
                                bgcolor: alpha(theme.palette.success.light, 0.1),
                                borderRadius: 2,
                                border: `1px solid ${alpha(theme.palette.success.main, 0.2)}`,
                                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)',
                                transition: 'all 0.3s ease',
                                '&:hover': {
                                  boxShadow: '0 6px 16px rgba(0, 0, 0, 0.08)',
                                  transform: 'translateY(-3px)',
                                }
                              }}
                            >
                              <AnimatedTypography 
                                component={HumanText}
                                variant="subtitle2" 
                                color="success.main" 
                                sx={{ 
                                  mb: 1, 
                                  display: 'flex', 
                                  alignItems: 'center',
                                  fontWeight: 600,
                                }}
                                style={useSpring({
                                  from: { opacity: 0, transform: 'translateY(5px)' },
                                  to: { opacity: 1, transform: 'translateY(0)' },
                                  delay: 300,
                                  config: { tension: 280, friction: 60 }
                                })}
                              >
                                <DataObjectIcon 
                                  fontSize="small" 
                                  sx={{ 
                                    mr: 1,
                                    animation: 'pulse 2s ease-in-out infinite',
                                    '@keyframes pulse': {
                                      '0%': {
                                        opacity: 0.8,
                                        transform: 'scale(1)',
                                      },
                                      '50%': {
                                        opacity: 1,
                                        transform: 'scale(1.1)',
                                      },
                                      '100%': {
                                        opacity: 0.8,
                                        transform: 'scale(1)',
                                      },
                                    },
                                  }} 
                                />
                                Column Characteristics
                              </AnimatedTypography>
                              <List dense>
                                {[
                                  {
                                    label: "Text columns",
                                    value: `${columns.filter(c => c.data_type.includes('CHAR') || c.data_type.includes('CLOB')).length} columns`
                                  },
                                  {
                                    label: "Numeric columns",
                                    value: `${columns.filter(c => c.data_type.includes('INT') || c.data_type.includes('DEC') || c.data_type.includes('NUM')).length} columns`
                                  },
                                  {
                                    label: "Date/time columns",
                                    value: `${columns.filter(c => c.data_type.includes('DATE') || c.data_type.includes('TIME')).length} columns`
                                  }
                                ].map((item, index) => (
                                  <AnimatedListItem
                                    key={index}
                                    style={useSpring({
                                      from: { opacity: 0, transform: 'translateX(-10px)' },
                                      to: { opacity: 1, transform: 'translateX(0)' },
                                      delay: 350 + (index * 100),
                                      config: { tension: 280, friction: 60 }
                                    })}
                                  >
                                    <ListItemText
                                      primary={<HumanText>{item.label}</HumanText>}
                                      secondary={<HumanText>{item.value}</HumanText>}
                                      primaryTypographyProps={{
                                        sx: {
                                          fontWeight: 500,
                                          color: theme.palette.success.dark
                                        }
                                      }}
                                    />
                                  </AnimatedListItem>
                                ))}
                              </List>
                            </AnimatedPaper>
                          </Grid>
                          <Grid item xs={12} md={6}>
                            <AnimatedPaper 
                              style={useSpring({
                                from: { opacity: 0, transform: 'translateY(20px)' },
                                to: { opacity: 1, transform: 'translateY(0)' },
                                delay: 300,
                                config: { tension: 280, friction: 60 }
                              })}
                              sx={{ 
                                p: 2, 
                                bgcolor: alpha(theme.palette.info.light, 0.1),
                                borderRadius: 2,
                                border: `1px solid ${alpha(theme.palette.info.main, 0.2)}`,
                                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)',
                                transition: 'all 0.3s ease',
                                '&:hover': {
                                  boxShadow: '0 6px 16px rgba(0, 0, 0, 0.08)',
                                  transform: 'translateY(-3px)',
                                }
                              }}
                            >
                              <AnimatedTypography 
                                component={HumanText}
                                variant="subtitle2" 
                                color="info.main" 
                                sx={{ 
                                  mb: 1, 
                                  display: 'flex', 
                                  alignItems: 'center',
                                  fontWeight: 600,
                                }}
                                style={useSpring({
                                  from: { opacity: 0, transform: 'translateY(5px)' },
                                  to: { opacity: 1, transform: 'translateY(0)' },
                                  delay: 350,
                                  config: { tension: 280, friction: 60 }
                                })}
                              >
                                <InfoIcon 
                                  fontSize="small" 
                                  sx={{ 
                                    mr: 1,
                                    animation: 'pulse 2s ease-in-out infinite 0.5s',
                                    '@keyframes pulse': {
                                      '0%': {
                                        opacity: 0.8,
                                        transform: 'scale(1)',
                                      },
                                      '50%': {
                                        opacity: 1,
                                        transform: 'scale(1.1)',
                                      },
                                      '100%': {
                                        opacity: 0.8,
                                        transform: 'scale(1)',
                                      },
                                    },
                                  }} 
                                />
                                Vectorization Potential
                              </AnimatedTypography>
                              <AnimatedTypography 
                                component={HumanText}
                                variant="body2" 
                                paragraph
                                style={useSpring({
                                  from: { opacity: 0, transform: 'translateY(10px)' },
                                  to: { opacity: 1, transform: 'translateY(0)' },
                                  delay: 400,
                                  config: { tension: 280, friction: 60 }
                                })}
                              >
                                This table contains {columns.filter(c => c.data_type.includes('CHAR') || c.data_type.includes('CLOB')).length} text columns that can be vectorized for semantic search.
                              </AnimatedTypography>
                              <AnimatedButton
                                fullWidth
                                variant="contained"
                                color="primary"
                                startIcon={<AutoAwesomeIcon />}
                                onClick={() => {
                                  if (onTableSelect) {
                                    onTableSelect(selectedTable.schema, selectedTable.table);
                                  }
                                }}
                                style={useSpring({
                                  from: { opacity: 0, transform: 'translateY(10px)' },
                                  to: { opacity: 1, transform: 'translateY(0)' },
                                  delay: 450,
                                  config: { tension: 280, friction: 60 }
                                })}
                                sx={{ 
                                  borderRadius: 2,
                                  background: 'linear-gradient(90deg, #0066B3, #19B5FE)',
                                  boxShadow: '0 4px 10px rgba(0, 102, 179, 0.2)',
                                  transition: 'all 0.3s ease',
                                  '&:hover': {
                                    transform: 'translateY(-2px)',
                                    boxShadow: '0 6px 14px rgba(0, 102, 179, 0.3)',
                                  },
                                  '&:active': {
                                    transform: 'translateY(0px)',
                                  },
                                  position: 'relative',
                                  overflow: 'hidden',
                                  '&::after': {
                                    content: '""',
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    width: '100%',
                                    height: '100%',
                                    background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent)',
                                    transform: 'translateX(-100%)',
                                  },
                                  '&:hover::after': {
                                    transform: 'translateX(100%)',
                                    transition: 'transform 0.6s ease',
                                  }
                                }}
                              >
                                Start Vectorization Process
                              </AnimatedButton>
                            </AnimatedPaper>
                          </Grid>
                        </Grid>
                      </AnimatedBox>
                    </>
                  )}
                </AnimatedBox>
              </>
            ) : (
              <AnimatedBox
                style={noSelectionAnimation}
                sx={{
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  height: '100%',
                  flexDirection: 'column',
                  p: 3,
                }}
              >
                <animated.div style={useSpring({
                  from: { opacity: 0, transform: 'scale(0.5) translateY(20px)' },
                  to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'scale(1) translateY(0px)' : 'scale(0.5) translateY(20px)' },
                  delay: 300,
                  config: { tension: 280, friction: 20 }
                })}>
                  <StorageIcon sx={{ 
                    fontSize: 80, 
                    color: alpha(theme.palette.primary.main, 0.2), 
                    mb: 2,
                    filter: 'drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1))',
                    animation: 'float 3s ease-in-out infinite',
                    '@keyframes float': {
                      '0%': {
                        transform: 'translateY(0px)',
                      },
                      '50%': {
                        transform: 'translateY(-10px)',
                      },
                      '100%': {
                        transform: 'translateY(0px)',
                      },
                    },
                  }} />
                </animated.div>
                <AnimatedTypography 
                  component={HumanText} 
                  variant="h6" 
                  style={{
                    background: 'linear-gradient(90deg, #0066B3 0%, #19B5FE 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    ...useSpring({
                      from: { opacity: 0, transform: 'translateY(20px)' },
                      to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(20px)' },
                      delay: 400,
                      config: { tension: 280, friction: 60 }
                    })
                  }}
                  sx={{ mb: 1, textAlign: 'center', fontWeight: 600 }}
                >
                  Select a Table
                </AnimatedTypography>
                <AnimatedTypography
                  component={HumanText}
                  variant="body2"
                  color="text.secondary"
                  style={useSpring({
                    from: { opacity: 0, transform: 'translateY(20px)' },
                    to: { opacity: animationsVisible ? 1 : 0, transform: animationsVisible ? 'translateY(0)' : 'translateY(20px)' },
                    delay: 500,
                    config: { tension: 280, friction: 60 }
                  })}
                  sx={{ maxWidth: 400, textAlign: 'center' }}
                >
                  Select a table from the schema browser to view its details and start the vectorization process.
                </AnimatedTypography>
              </AnimatedBox>
            )}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default SchemaExplorer;