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

// Flow types for the Developer environment
export interface FlowNode {
  id: string;
  type: string;
  data: {
    label?: string;
    params?: Record<string, any>;
    results?: any[];
    [key: string]: any;
  };
  position: {
    x: number;
    y: number;
  };
}

export interface FlowEdge {
  id: string;
  source: string;
  target: string;
  type?: string;
  animated?: boolean;
  markerEnd?: any;
}

export interface Flow {
  id?: string;
  name: string;
  description: string;
  nodes: FlowNode[];
  edges: FlowEdge[];
  created_at?: string;
  updated_at?: string;
}

export interface RunFlowRequest {
  flow: Flow;
}

export interface RunFlowResponse {
  success: boolean;
  results: any[];
  execution_time: number;
  generated_code: string;
}

export interface SaveFlowRequest {
  flow: Flow;
}

export interface SaveFlowResponse {
  success: boolean;
  flow_id: string;
  message: string;
}

export interface ListFlowsResponse {
  flows: Flow[];
}

export interface VectorDataPoint {
  id: string;
  content: string;
  metadata: Record<string, any>;
  vector: number[];
  reduced_vector: number[];
}

export interface GetVectorsRequest {
  tableName: string;
  filter?: Record<string, any>;
  maxPoints?: number;
  page?: number;
  pageSize?: number;
  clusteringAlgorithm?: 'kmeans' | 'dbscan' | 'hdbscan';
  dimensionalityReduction?: 'tsne' | 'umap' | 'pca';
}

export interface GetVectorsResponse {
  vectors: VectorDataPoint[];
  total_count: number;
  page: number;
  page_size: number;
  total_pages: number;
}

// Debug types
export type DebugNodeStatus = 'not_executed' | 'executing' | 'completed' | 'error';
export type DebugSessionStatus = 'ready' | 'running' | 'paused' | 'completed' | 'error';
export type DebugStepType = 'step' | 'step_over' | 'continue' | 'reset';

export interface DebugBreakpoint {
  node_id: string;
  enabled: boolean;
  condition?: string;
}

export interface DebugNodeData {
  node_id: string;
  input_data?: any;
  output_data?: any;
  execution_time?: number;
  status: DebugNodeStatus;
  error?: string;
}

export interface DebugSession {
  session_id: string;
  flow_id?: string;
  breakpoints: DebugBreakpoint[];
  current_node_id?: string;
  status: DebugSessionStatus;
  node_data: Record<string, DebugNodeData>;
  variables: Record<string, any>;
  created_at?: string;
  updated_at?: string;
}

export interface CreateDebugSessionRequest {
  flow: Flow;
  breakpoints?: DebugBreakpoint[];
}

export interface CreateDebugSessionResponse {
  session_id: string;
  status: string;
  message: string;
}

export interface DebugStepRequest {
  session_id: string;
  step_type: DebugStepType;
}

export interface DebugStepResponse {
  session: DebugSession;
  node_output?: any;
  execution_time: number;
  message: string;
}

export interface SetBreakpointRequest {
  session_id: string;
  breakpoint: DebugBreakpoint;
}

export interface SetBreakpointResponse {
  success: boolean;
  message: string;
}

export interface GetVariablesRequest {
  session_id: string;
  variable_names?: string[];
}

export interface GetVariablesResponse {
  variables: Record<string, any>;
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

// Developer service
export const developerService = {
  runFlow: (flow: Flow) => 
    apiClient.post<RunFlowResponse>('/developer/run', { flow }),
  
  generateCode: (flow: Flow) => 
    apiClient.post<{ code: string }>('/developer/generate-code', flow),
  
  saveFlow: (flow: Flow) => 
    apiClient.post<SaveFlowResponse>('/developer/flows', { flow }),
  
  listFlows: () => 
    apiClient.get<ListFlowsResponse>('/developer/flows'),
  
  getFlow: (flowId: string) => 
    apiClient.get<Flow>(`/developer/flows/${flowId}`),
  
  deleteFlow: (flowId: string) => 
    apiClient.delete(`/developer/flows/${flowId}`),
    
  getVectors: (request: GetVectorsRequest) => 
    apiClient.post<GetVectorsResponse>('/developer/vectors', request),
    
  // Debug methods
  createDebugSession: (flow: Flow, breakpoints?: DebugBreakpoint[]) => 
    apiClient.post<CreateDebugSessionResponse>('/developer/debug/sessions', { 
      flow, 
      breakpoints 
    }),
    
  getDebugSession: (sessionId: string) => 
    apiClient.get<DebugSession>(`/developer/debug/sessions/${sessionId}`),
    
  deleteDebugSession: (sessionId: string) => 
    apiClient.delete(`/developer/debug/sessions/${sessionId}`),
    
  stepDebugSession: (sessionId: string, stepType: DebugStepType) => 
    apiClient.post<DebugStepResponse>('/developer/debug/step', {
      session_id: sessionId,
      step_type: stepType
    }),
    
  setBreakpoint: (sessionId: string, breakpoint: DebugBreakpoint) => 
    apiClient.post<SetBreakpointResponse>('/developer/debug/breakpoints', {
      session_id: sessionId,
      breakpoint
    }),
    
  getVariables: (sessionId: string, variableNames?: string[]) => 
    apiClient.post<GetVariablesResponse>('/developer/debug/variables', {
      session_id: sessionId,
      variable_names: variableNames
    }),
};