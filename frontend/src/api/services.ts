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

// Analytics service
export interface RecentQuery {
  id: number;
  query: string;
  timestamp: string;
  results_count: number;
  execution_time: number;
}

export interface PerformanceStats {
  name: string;
  queries: number;
  avgTime: number;
}

export interface PerformanceComparison {
  name: string;
  CPU: number;
  GPU: number;
  TensorRT: number;
}

export const analyticsService = {
  getRecentQueries: () => 
    apiClient.get<RecentQuery[]>('/analytics/recent-queries'),
  
  getPerformanceStats: () => 
    apiClient.get<PerformanceStats[]>('/analytics/performance'),
  
  getQueryCount: () => 
    apiClient.get<{count: number}>('/analytics/query-count'),
  
  getAverageResponseTime: () => 
    apiClient.get<{avgTime: number}>('/analytics/response-time'),
    
  getPerformanceComparison: () => 
    apiClient.get<PerformanceComparison[]>('/analytics/performance-comparison'),
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

// GitHub Export Options
export interface GitHubExportOptions {
  flowId: string;
  code: string;
  repository: string;
  path: string;
  message: string;
  description: string;
}

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
    
  // New methods for deployment and GitHub export
  deployFlow: (flowId: string) => 
    apiClient.post<APIResponse>(`/developer/flows/${flowId}/deploy`),
    
  exportToGitHub: (options: GitHubExportOptions) => 
    apiClient.post<APIResponse>('/developer/export/github', options),
    
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

// Types for the Reasoning API
export interface ReasoningStep {
  step_number: number;
  description: string;
  reasoning: string;
  confidence: number;
  references: {
    id: string;
    relevance: number;
    content?: string;
  }[];
}

export interface ReasoningPathRequest {
  query: string;
  table_name?: string;
  document_id?: string;
  include_content?: boolean;
  max_steps?: number;
}

export interface ReasoningPathResponse {
  path_id: string;
  query: string;
  steps: ReasoningStep[];
  final_result?: string;
  metadata: Record<string, any>;
  execution_time: number;
}

export interface TransformationStage {
  stage_number: number;
  name: string;
  description: string;
  input_type: string;
  output_type: string;
  duration_ms: number;
  metadata: Record<string, any>;
  sample_output?: any;
}

export interface TransformationRequest {
  document_id: string;
  table_name?: string;
  include_intermediate?: boolean;
  track_only?: boolean;
}

export interface TransformationResponse {
  transformation_id: string;
  document_id: string;
  stages: TransformationStage[];
  metadata: Record<string, any>;
  execution_time: number;
}

export interface ValidationRequest {
  reasoning_path_id: string;
  validation_types?: string[];
}

export interface ValidationResponse {
  validation_id: string;
  reasoning_path_id: string;
  results: Record<string, any>;
  score: number;
  suggestions: string[];
  execution_time: number;
}

export interface MetricsRequest {
  document_id?: string;
  table_name?: string;
  metric_types?: string[];
}

export interface MetricsResponse {
  metrics_id: string;
  document_id?: string;
  metrics: Record<string, any>;
  execution_time: number;
}

export interface FeedbackRequest {
  query: string;
  document_id?: string;
  reasoning_path_id?: string;
  feedback_type: string;
  feedback_content: string;
  rating?: number;
  user_id?: string;
}

export interface FeedbackResponse {
  feedback_id: string;
  status: string;
  message: string;
}

export interface FingerprintRequest {
  document_id: string;
  table_name?: string;
  include_lineage?: boolean;
}

export interface FingerprintResponse {
  fingerprint_id: string;
  document_id: string;
  signatures: Record<string, any>;
  lineage?: Record<string, any>;
  execution_time: number;
}

export interface ReasoningStatusResponse {
  available: boolean;
  features: Record<string, boolean>;
  version?: string;
}

// Reasoning service
export const reasoningService = {
  trackReasoningPath: (request: ReasoningPathRequest) => 
    apiClient.post<ReasoningPathResponse>('/reasoning/track', request),
  
  trackTransformation: (request: TransformationRequest) => 
    apiClient.post<TransformationResponse>('/reasoning/transformation', request),
  
  validateReasoning: (request: ValidationRequest) => 
    apiClient.post<ValidationResponse>('/reasoning/validate', request),
  
  calculateMetrics: (request: MetricsRequest) => 
    apiClient.post<MetricsResponse>('/reasoning/metrics', request),
  
  submitFeedback: (request: FeedbackRequest) => 
    apiClient.post<FeedbackResponse>('/reasoning/feedback', request),
  
  getFingerprint: (request: FingerprintRequest) => 
    apiClient.post<FingerprintResponse>('/reasoning/fingerprint', request),
  
  getStatus: () => 
    apiClient.get<APIResponse>('/reasoning/status'),
};

// Types for the Data Pipeline API
export interface CreatePipelineRequest {
  connection_id?: string;
}

export interface CreatePipelineResponse {
  pipeline_id: string;
  status: string;
  message: string;
}

export interface RegisterDataSourceRequest {
  pipeline_id: string;
  schema_name: string;
  table_name: string;
  include_sample?: boolean;
  sample_size?: number;
  custom_metadata?: Record<string, any>;
}

export interface RegisterDataSourceResponse {
  source_id: string;
  pipeline_id: string;
  schema_name: string;
  table_name: string;
  status: string;
  message: string;
}

export interface RegisterIntermediateStageRequest {
  pipeline_id: string;
  stage_name: string;
  stage_description: string;
  source_id: string;
  column_mapping: Record<string, string[]>;
  data_sample?: Record<string, any>[];
  processing_metadata?: Record<string, any>;
}

export interface RegisterIntermediateStageResponse {
  stage_id: string;
  pipeline_id: string;
  stage_name: string;
  source_id: string;
  status: string;
  message: string;
}

export interface RegisterVectorRequest {
  pipeline_id: string;
  source_id: string;
  model_name: string;
  vector_dimensions: number;
  vector_sample?: number[];
  original_text?: string;
  processing_metadata?: Record<string, any>;
}

export interface RegisterVectorResponse {
  vector_id: string;
  pipeline_id: string;
  source_id: string;
  model_name: string;
  status: string;
  message: string;
}

export interface RegisterTransformationRuleRequest {
  pipeline_id: string;
  rule_name: string;
  rule_description: string;
  input_columns: string[];
  output_columns: string[];
  transformation_type: string;
  transformation_params: Record<string, any>;
}

export interface RegisterTransformationRuleResponse {
  rule_id: string;
  pipeline_id: string;
  rule_name: string;
  status: string;
  message: string;
}

export interface GetPipelineRequest {
  pipeline_id: string;
  source_id?: string;
}

export interface GetPipelineResponse {
  pipeline_id: string;
  data_sources: Record<string, any>;
  intermediate_stages: Record<string, any>;
  vector_representations: Record<string, any>;
  transformation_rules: Record<string, any>;
  created_at: number;
}

export interface GetDataLineageRequest {
  pipeline_id: string;
  vector_id: string;
}

export interface GetDataLineageResponse {
  vector_id: string;
  vector_data: Record<string, any>;
  source_data: Record<string, any>;
  transformation_stages: Record<string, any>[];
  created_at: number;
}

export interface GetReverseMapRequest {
  pipeline_id: string;
  vector_id: string;
  similarity_threshold?: number;
}

export interface GetReverseMapResponse {
  vector_id: string;
  source_data: Record<string, any>;
  similar_vectors: Record<string, any>[];
  threshold: number;
  created_at: number;
}

export interface ListPipelinesResponse {
  pipelines: Record<string, any>[];
  count: number;
}

// Data Pipeline Service
export const dataPipelineService = {
  createPipeline: (request: CreatePipelineRequest) =>
    apiClient.post<CreatePipelineResponse>('/data-pipeline/create', request),
  
  registerDataSource: (request: RegisterDataSourceRequest) =>
    apiClient.post<RegisterDataSourceResponse>('/data-pipeline/register-source', request),
  
  registerIntermediateStage: (request: RegisterIntermediateStageRequest) =>
    apiClient.post<RegisterIntermediateStageResponse>('/data-pipeline/register-intermediate', request),
  
  registerVector: (request: RegisterVectorRequest) =>
    apiClient.post<RegisterVectorResponse>('/data-pipeline/register-vector', request),
  
  registerTransformationRule: (request: RegisterTransformationRuleRequest) =>
    apiClient.post<RegisterTransformationRuleResponse>('/data-pipeline/register-rule', request),
  
  getPipeline: (request: GetPipelineRequest) =>
    apiClient.post<GetPipelineResponse>('/data-pipeline/get', request),
  
  getDataLineage: (request: GetDataLineageRequest) =>
    apiClient.post<GetDataLineageResponse>('/data-pipeline/lineage', request),
  
  getReverseMap: (request: GetReverseMapRequest) =>
    apiClient.post<GetReverseMapResponse>('/data-pipeline/reverse-map', request),
  
  listPipelines: () =>
    apiClient.get<ListPipelinesResponse>('/data-pipeline/list'),
  
  getStatus: () =>
    apiClient.get<APIResponse>('/data-pipeline/status'),
};

// Types for Vector Operations API
export interface CreateVectorRequest {
  pipeline_id: string;
  source_id: string;
  table_name: string;
  schema_name?: string;
  model_name?: string;
  vector_dimensions?: number;
  normalize_vectors?: boolean;
  chunking_strategy?: string;
  chunk_size?: number;
  chunk_overlap?: number;
  max_records?: number;
  filter_condition?: string;
  embedding_type?: string;
  pal_batch_size?: number;
  use_pal_service?: boolean;
}

export interface CreateVectorResponse {
  vector_id: string;
  table_name: string;
  vector_count: number;
  model_name: string;
  dimensions: number;
  processing_time: number;
  status: string;
  message: string;
}

export interface VectorInfoRequest {
  vector_id: string;
}

export interface VectorInfoResponse {
  vector_id: string;
  table_name: string;
  vector_count: number;
  model_name: string;
  dimensions: number;
  sample_vector?: number[];
  metadata: Record<string, any>;
  created_at?: string;
}

export interface BatchEmbeddingRequest {
  texts: string[];
  model_name?: string;
  embedding_type?: string;
  use_pal_service?: boolean;
}

export interface BatchEmbeddingResponse {
  embeddings: number[][];
  dimensions: number;
  processing_time: number;
  tokens_processed: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  dimensions: number;
  description: string;
  performance: 'fast' | 'medium' | 'slow';
  quality: 'basic' | 'good' | 'excellent';
  embedding_types: string[];
  pal_service?: boolean;
}

export interface ModelsResponse {
  models: ModelInfo[];
  pal_available: boolean;
  vector_engine_available: boolean;
  recommended_model: string;
  error?: string;
}

// Vector Operations service
export const vectorOperationsService = {
  createVectors: (request: CreateVectorRequest) =>
    apiClient.post<CreateVectorResponse>('/vector-operations/create', request),
  
  getVectorInfo: (request: VectorInfoRequest) =>
    apiClient.post<VectorInfoResponse>('/vector-operations/info', request),
  
  batchEmbed: (request: BatchEmbeddingRequest) =>
    apiClient.post<BatchEmbeddingResponse>('/vector-operations/batch-embed', request),
  
  listModels: () =>
    apiClient.get<ModelsResponse>('/vector-operations/models'),
};