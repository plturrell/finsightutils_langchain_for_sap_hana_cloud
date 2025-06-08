/**
 * humanLanguage.ts
 * 
 * This module replaces technical terminology with human-friendly language
 * throughout the application, following Apple's principles of focusing
 * on benefits rather than implementation details.
 */

// Technical term to human-friendly term mapping
export const humanTerms: Record<string, string> = {
  // Core concepts
  "Vector Search": "Meaning Search",
  "Vectorstore": "Knowledge Library",
  "Embeddings": "Understanding",
  "Similarity": "Relevance",
  "GPU Acceleration": "Speed Boost",
  "TensorRT": "Performance Engine",
  "Database": "Storage",
  "API": "Connection",
  "Benchmark": "Speed Test",
  "Metadata": "Details",
  "Query": "Question",
  "Document": "Information",
  "Precision": "Accuracy",
  "Configuration": "Preferences",
  "Batching": "Processing",
  "Parameters": "Settings",
  "Vector": "Pattern",
  "Dimension": "Complexity",
  
  // Technical operations
  "Retrieve": "Find",
  "Index": "Organize",
  "Configure": "Set up",
  "Optimize": "Improve",
  "Cache": "Remember",
  "Deployment": "Setup",
  "Initialize": "Start",
  "Authentication": "Security",
  
  // UI elements
  "Threshold": "Sensitivity",
  "Score": "Match Quality",
  "Filter": "Narrow Down",
  "Cosine": "Matching",
  "Fetch": "Retrieve",
  "MMR": "Diversity",
  "Lambda": "Balance",
  "Persist": "Save",
  "Toggle": "Switch",
  "Collapse": "Hide",
  "Expand": "Show",
  
  // Technical parameters
  "Batch Size": "Processing Amount",
  "fp16": "Fast Mode",
  "fp32": "Precise Mode",
  "int8": "Efficient Mode",
  "CUDA": "Graphics Processor",
  "CPU": "Computer Processor",
  "HNSW": "Fast Search",
  "Tensor": "Pattern Matcher",
  
  // SAP HANA specific
  "HANA": "Database",
  "SAP HANA Cloud": "Cloud Storage",
  "Table Name": "Storage Location",
  "Column": "Data Field",
  "Schema": "Organization",
  "Connection String": "Access Path",
};

/**
 * Convert technical language to human-friendly language
 * 
 * @param text The technical text to humanize
 * @returns Human-friendly text
 */
export function humanize(text: string): string {
  if (!text) return text;
  
  let humanized = text;
  
  // Replace each technical term with its human-friendly equivalent
  Object.entries(humanTerms).forEach(([technical, human]) => {
    // Create a regex that matches the technical term as a whole word
    // This prevents replacing parts of words
    const regex = new RegExp(`\\b${technical}\\b`, 'gi');
    humanized = humanized.replace(regex, human);
  });
  
  return humanized;
}

/**
 * Humanize object property names and string values
 * 
 * @param obj The object to humanize
 * @returns Object with humanized property names and string values
 */
export function humanizeObject<T extends Record<string, any>>(obj: T): Record<string, any> {
  const result: Record<string, any> = {};
  
  Object.entries(obj).forEach(([key, value]) => {
    // Humanize the key
    const humanKey = humanize(key);
    
    // Humanize the value if it's a string
    if (typeof value === 'string') {
      result[humanKey] = humanize(value);
    } 
    // Recursively humanize nested objects
    else if (value && typeof value === 'object' && !Array.isArray(value)) {
      result[humanKey] = humanizeObject(value);
    }
    // Handle arrays - humanize string elements
    else if (Array.isArray(value)) {
      result[humanKey] = value.map(item => 
        typeof item === 'string' ? humanize(item) : item
      );
    }
    // Keep other values as is
    else {
      result[humanKey] = value;
    }
  });
  
  return result;
}

/**
 * React component props to humanize text
 */
export interface HumanizeProps {
  text: string;
}

/**
 * Format a technical score (0-1) as a human-friendly percentage
 * 
 * @param score The technical score (0-1)
 * @returns A formatted percentage string
 */
export function formatScore(score: number): string {
  return `${Math.round(score * 100)}%`;
}

/**
 * Convert technical file size to human-readable format
 * 
 * @param bytes File size in bytes
 * @returns Human-readable file size
 */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

/**
 * Simplify technical date formats to human-friendly ones
 * 
 * @param dateString A date string in any format
 * @returns A human-friendly date string
 */
export function formatDate(dateString: string): string {
  try {
    const date = new Date(dateString);
    
    // Check if date is valid
    if (isNaN(date.getTime())) {
      return dateString;
    }
    
    // Get current date for comparison
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    
    // Format based on how recent the date is
    if (date >= today) {
      return `Today at ${date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}`;
    } else if (date >= yesterday) {
      return `Yesterday at ${date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}`;
    } else if (now.getFullYear() === date.getFullYear()) {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    } else {
      return date.toLocaleDateString([], { year: 'numeric', month: 'short', day: 'numeric' });
    }
  } catch (e) {
    // If anything goes wrong, return the original string
    return dateString;
  }
}