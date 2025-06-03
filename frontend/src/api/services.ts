import apiClient from './client';

// Types
export interface Document {
  page_content: string;
  metadata: Record<string, any>;
}

export interface SearchResult {
  document: Document;
  score: number;
}

export interface SearchResponse {
  results: SearchResult[];
}

export interface APIResponse {
  success: boolean;
  message: string;
  data?: any;
}

export interface BenchmarkRequest {
  texts?: string[];
  count?: number;
  batch_size?: number;
  query?: string;
  k?: number;
  iterations?: number;
  model_name?: string;
  precision?: string;
  batch_sizes?: number[];
  input_length?: number;
}

export interface BenchmarkResult {
  [key: string]: any;
}

export interface GPUInfo {
  gpu_available: boolean;
  device_count: number;
  devices: {
    name: string;
    memory_total: number;
    memory_used: number;
    memory_free: number;
    utilization: number;
  }[];
}

// Health service
export const healthService = {
  check: () => apiClient.get('/health'),
};

// Vector Store service
export const vectorStoreService = {
  addTexts: (texts: string[], metadatas?: Record<string, any>[]) => 
    apiClient.post<APIResponse>('/texts', { texts, metadatas }),
  
  query: (query: string, k = 4, filter?: Record<string, any>) => 
    apiClient.post<SearchResponse>('/query', { query, k, filter }),
  
  queryByVector: (embedding: number[], k = 4, filter?: Record<string, any>) => 
    apiClient.post<SearchResponse>('/query/vector', { embedding, k, filter }),
  
  mmrQuery: (query: string, k = 4, fetch_k = 20, lambda_mult = 0.5, filter?: Record<string, any>) => 
    apiClient.post<SearchResponse>('/query/mmr', { query, k, fetch_k, lambda_mult, filter }),
  
  delete: (filter: Record<string, any>) => 
    apiClient.post<APIResponse>('/delete', { filter }),
};

// Benchmark service
export const benchmarkService = {
  status: () => apiClient.get('/benchmark/status'),
  
  gpuInfo: () => apiClient.get<GPUInfo>('/benchmark/gpu_info'),
  
  embedding: (request: BenchmarkRequest) => 
    apiClient.post<BenchmarkResult>('/benchmark/embedding', request),
  
  vectorSearch: (request: BenchmarkRequest) => 
    apiClient.post<BenchmarkResult>('/benchmark/search', request),
  
  tensorrt: (request: BenchmarkRequest) => 
    apiClient.post<BenchmarkResult>('/benchmark/tensorrt', request),
  
  compareEmbeddings: () => 
    apiClient.post<BenchmarkResult>('/benchmark/compare_embeddings'),
};

// GPU service
export const gpuService = {
  info: () => apiClient.get('/gpu/info'),
};

// Analytics service
export const analyticsService = {
  getRecentQueries: () => apiClient.get('/analytics/recent_queries'),
  getPerformanceStats: () => apiClient.get('/analytics/performance'),
};