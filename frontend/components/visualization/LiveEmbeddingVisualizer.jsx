import React, { useState, useEffect, useRef, useCallback } from 'react';
import EmbeddingVisualizerReact from './EmbeddingVisualizerReact';

/**
 * LiveEmbeddingVisualizer - Enhanced embedding visualization with real-time updates,
 * business value metrics, and financial domain-specific features.
 * 
 * This component extends the base EmbeddingVisualizerReact with:
 * - Real-time updates for embeddings as they change
 * - Business value translations of technical metrics
 * - Financial domain-specific color coding and visualizations
 * - Before/after comparison view for showing embedding improvements
 * - Shareable results dashboard for business stakeholders
 */
const LiveEmbeddingVisualizer = ({
  // Base visualization props
  points = [],
  metadata = null,
  similarities = null,
  queryPoint = null,
  clusters = null,
  width = '100%',
  height = '500px',
  method = '3d',
  colorScheme = 'similarity',
  showLabels = true,
  showAxes = true,
  showGrid = true,
  colorblindMode = false,
  autoRotate = false,
  onPointSelected = null,
  className = '',
  style = {},

  // Live update props
  liveUpdateEnabled = false,
  updateInterval = 2000,
  dataSource = null,
  animateUpdates = true,
  
  // Business value props
  showBusinessMetrics = true,
  businessContext = 'financial',
  confidenceThreshold = 0.8,
  
  // Financial domain props
  financialCategories = null,
  showFinancialLegend = true,
  highlightSentiment = false,
  
  // Comparison view props
  comparisonEnabled = false,
  comparisonPoints = null,
  comparisonLabel = 'After Optimization',
  baselineLabel = 'Before Optimization',
  
  // Sharing props
  enableSharing = false,
  reportTitle = 'Embedding Analysis',
  reportDescription = '',
  includeMetadata = true
}) => {
  // State for managing live data
  const [currentPoints, setCurrentPoints] = useState(points);
  const [currentMetadata, setCurrentMetadata] = useState(metadata);
  const [currentSimilarities, setCurrentSimilarities] = useState(similarities);
  const [currentQueryPoint, setCurrentQueryPoint] = useState(queryPoint);
  const [currentClusters, setCurrentClusters] = useState(clusters);
  
  // State for business metrics
  const [businessMetrics, setBusinessMetrics] = useState({
    confidenceScore: 0,
    relevanceScore: 0,
    financialInsights: [],
    improvementPercent: 0,
    categoryCoverage: {}
  });
  
  // State for animation and transitions
  const [isAnimating, setIsAnimating] = useState(false);
  const [transitionProgress, setTransitionProgress] = useState(1);
  
  // State for comparison view
  const [showComparison, setShowComparison] = useState(comparisonEnabled && comparisonPoints);
  const [activeView, setActiveView] = useState('current'); // 'current', 'comparison', 'split'
  
  // State for sharing
  const [shareUrl, setShareUrl] = useState('');
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  
  // Refs
  const visualizerRef = useRef(null);
  const comparisonVisualizerRef = useRef(null);
  const updateTimerRef = useRef(null);
  const previousPointsRef = useRef(points);

  // Initialize component
  useEffect(() => {
    // Set initial data
    setCurrentPoints(points);
    setCurrentMetadata(metadata);
    setCurrentSimilarities(similarities);
    setCurrentQueryPoint(queryPoint);
    setCurrentClusters(clusters);
    
    // Calculate initial business metrics
    calculateBusinessMetrics(points, similarities, metadata, clusters);
    
    // Start live updates if enabled
    if (liveUpdateEnabled && dataSource) {
      startLiveUpdates();
    }
    
    return () => {
      // Clean up timer on unmount
      if (updateTimerRef.current) {
        clearInterval(updateTimerRef.current);
      }
    };
  }, []);
  
  // Handle props changes
  useEffect(() => {
    if (!isAnimating) {
      setCurrentPoints(points);
      setCurrentMetadata(metadata);
      setCurrentSimilarities(similarities);
      setCurrentQueryPoint(queryPoint);
      setCurrentClusters(clusters);
      
      calculateBusinessMetrics(points, similarities, metadata, clusters);
    }
    
    previousPointsRef.current = points;
  }, [points, metadata, similarities, queryPoint, clusters, isAnimating]);
  
  // Handle comparison mode changes
  useEffect(() => {
    setShowComparison(comparisonEnabled && comparisonPoints);
  }, [comparisonEnabled, comparisonPoints]);
  
  // Start live updates
  const startLiveUpdates = () => {
    if (updateTimerRef.current) {
      clearInterval(updateTimerRef.current);
    }
    
    updateTimerRef.current = setInterval(() => {
      if (dataSource && typeof dataSource === 'function') {
        fetchLiveData();
      }
    }, updateInterval);
  };
  
  // Fetch live data from data source
  const fetchLiveData = async () => {
    try {
      const newData = await dataSource();
      
      if (newData) {
        const {
          points: newPoints,
          metadata: newMetadata,
          similarities: newSimilarities,
          queryPoint: newQueryPoint,
          clusters: newClusters
        } = newData;
        
        if (newPoints && newPoints.length > 0) {
          if (animateUpdates) {
            animatePointTransition(newPoints, newSimilarities, newMetadata, newQueryPoint, newClusters);
          } else {
            setCurrentPoints(newPoints);
            setCurrentMetadata(newMetadata || null);
            setCurrentSimilarities(newSimilarities || null);
            setCurrentQueryPoint(newQueryPoint || null);
            setCurrentClusters(newClusters || null);
            
            calculateBusinessMetrics(newPoints, newSimilarities, newMetadata, newClusters);
          }
        }
      }
    } catch (error) {
      console.error('Error fetching live embedding data:', error);
    }
  };
  
  // Animate transition between point sets
  const animatePointTransition = (newPoints, newSimilarities, newMetadata, newQueryPoint, newClusters) => {
    const oldPoints = currentPoints;
    
    // Only animate if we have the same number of points or can map between them
    if (oldPoints.length !== newPoints.length) {
      // If points count doesn't match, just update without animation
      setCurrentPoints(newPoints);
      setCurrentMetadata(newMetadata || null);
      setCurrentSimilarities(newSimilarities || null);
      setCurrentQueryPoint(newQueryPoint || null);
      setCurrentClusters(newClusters || null);
      
      calculateBusinessMetrics(newPoints, newSimilarities, newMetadata, newClusters);
      return;
    }
    
    setIsAnimating(true);
    setTransitionProgress(0);
    
    // Animation frames
    const frames = 30;
    let frame = 0;
    
    const animate = () => {
      if (frame <= frames) {
        const progress = frame / frames;
        setTransitionProgress(progress);
        
        // Interpolate points
        const interpolatedPoints = oldPoints.map((oldPoint, i) => {
          const newPoint = newPoints[i];
          return oldPoint.map((value, dim) => {
            const newValue = newPoint[dim] || 0;
            return value + (newValue - value) * progress;
          });
        });
        
        // Interpolate similarities if available
        let interpolatedSimilarities = null;
        if (currentSimilarities && newSimilarities) {
          interpolatedSimilarities = currentSimilarities.map((oldSim, i) => {
            const newSim = newSimilarities[i] || 0;
            return oldSim + (newSim - oldSim) * progress;
          });
        }
        
        // Update points
        setCurrentPoints(interpolatedPoints);
        if (interpolatedSimilarities) {
          setCurrentSimilarities(interpolatedSimilarities);
        }
        
        frame++;
        requestAnimationFrame(animate);
      } else {
        // Final update with exact new values
        setCurrentPoints(newPoints);
        setCurrentMetadata(newMetadata || null);
        setCurrentSimilarities(newSimilarities || null);
        setCurrentQueryPoint(newQueryPoint || null);
        setCurrentClusters(newClusters || null);
        
        calculateBusinessMetrics(newPoints, newSimilarities, newMetadata, newClusters);
        setIsAnimating(false);
        setTransitionProgress(1);
      }
    };
    
    requestAnimationFrame(animate);
  };
  
  // Calculate business metrics from technical data
  const calculateBusinessMetrics = (points, similarities, metadata, clusters) => {
    // Initialize metrics
    const metrics = {
      confidenceScore: 0,
      relevanceScore: 0,
      financialInsights: [],
      improvementPercent: 0,
      categoryCoverage: {}
    };
    
    // Skip calculation if no data
    if (!points || points.length === 0) {
      setBusinessMetrics(metrics);
      return;
    }
    
    // Calculate confidence score from similarities
    if (similarities && similarities.length > 0) {
      // Average of top 5 similarities (or fewer if less available)
      const sortedSimilarities = [...similarities].sort((a, b) => b - a);
      const topSimilarities = sortedSimilarities.slice(0, Math.min(5, sortedSimilarities.length));
      metrics.confidenceScore = topSimilarities.reduce((sum, sim) => sum + sim, 0) / topSimilarities.length;
    }
    
    // Calculate relevance score
    if (similarities && similarities.length > 0) {
      // Count of similarities above confidence threshold
      const relevantCount = similarities.filter(sim => sim >= confidenceThreshold).length;
      metrics.relevanceScore = relevantCount / similarities.length;
    }
    
    // Extract financial insights from metadata
    if (metadata && metadata.length > 0) {
      // Find insights in metadata
      const insightPhrases = [
        'increase', 'decrease', 'growth', 'decline', 'revenue', 
        'profit', 'margin', 'dividend', 'earnings', 'debt'
      ];
      
      const insights = [];
      
      metadata.forEach((meta, i) => {
        if (!meta || !meta.title) return;
        
        // Check for insight phrases in title or content
        const text = (meta.title + ' ' + (meta.content || '')).toLowerCase();
        insightPhrases.forEach(phrase => {
          if (text.includes(phrase) && similarities && similarities[i] >= confidenceThreshold) {
            insights.push({
              text: meta.title,
              similarity: similarities[i],
              phrase: phrase
            });
          }
        });
      });
      
      // Sort by similarity and take top insights
      metrics.financialInsights = insights
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 3);
    }
    
    // Calculate improvement compared to previous data
    if (previousPointsRef.current && previousPointsRef.current.length > 0 && similarities) {
      const prevSimilarities = similarities;
      const currentAvgSim = similarities.reduce((sum, sim) => sum + sim, 0) / similarities.length;
      const prevAvgSim = prevSimilarities.reduce((sum, sim) => sum + sim, 0) / prevSimilarities.length;
      
      if (prevAvgSim > 0) {
        metrics.improvementPercent = (currentAvgSim - prevAvgSim) / prevAvgSim * 100;
      }
    }
    
    // Calculate category coverage if financial categories are provided
    if (financialCategories && clusters) {
      // Create mapping of clusters to categories
      const categoryMap = {};
      
      clusters.forEach((cluster, i) => {
        if (cluster === undefined || cluster === null) return;
        
        const categoryIndex = cluster % financialCategories.length;
        const category = financialCategories[categoryIndex];
        
        if (!categoryMap[category]) {
          categoryMap[category] = 0;
        }
        
        categoryMap[category]++;
      });
      
      // Calculate percentage coverage for each category
      Object.keys(categoryMap).forEach(category => {
        categoryMap[category] = categoryMap[category] / clusters.length;
      });
      
      metrics.categoryCoverage = categoryMap;
    }
    
    setBusinessMetrics(metrics);
  };
  
  // Generate shareable report
  const generateShareableReport = async () => {
    setIsGeneratingReport(true);
    
    try {
      // Capture visualizer image
      let imageUrl = null;
      if (visualizerRef.current) {
        const visInstance = visualizerRef.current;
        if (visInstance.exportImage) {
          imageUrl = await visInstance.exportImage();
        }
      }
      
      // Create report data
      const reportData = {
        title: reportTitle,
        description: reportDescription,
        timestamp: new Date().toISOString(),
        metrics: businessMetrics,
        image: imageUrl,
        pointCount: currentPoints?.length || 0,
        method: method,
        metadata: includeMetadata ? currentMetadata : null
      };
      
      // Generate shareable URL (in real implementation, this would save to server)
      // For demo, we'll just encode in base64
      const encodedReport = btoa(JSON.stringify(reportData));
      const shareLink = `${window.location.origin}/share?report=${encodedReport}`;
      
      setShareUrl(shareLink);
    } catch (error) {
      console.error('Error generating shareable report:', error);
    } finally {
      setIsGeneratingReport(false);
    }
  };
  
  // Translate technical metrics to business language
  const getBusinessDescription = useCallback(() => {
    const { confidenceScore, relevanceScore, improvementPercent } = businessMetrics;
    
    let description = '';
    
    // Confidence description
    if (confidenceScore >= 0.9) {
      description += 'Very high confidence match. ';
    } else if (confidenceScore >= 0.7) {
      description += 'Strong confidence in these results. ';
    } else if (confidenceScore >= 0.5) {
      description += 'Moderate confidence in these results. ';
    } else {
      description += 'Low confidence, consider refining your query. ';
    }
    
    // Relevance description
    if (relevanceScore >= 0.8) {
      description += 'Most documents are highly relevant. ';
    } else if (relevanceScore >= 0.5) {
      description += 'About half of documents are strongly relevant. ';
    } else if (relevanceScore >= 0.2) {
      description += 'A few key documents found. ';
    } else {
      description += 'Limited relevant results found. ';
    }
    
    // Improvement description
    if (Math.abs(improvementPercent) > 1) {
      if (improvementPercent > 20) {
        description += 'Dramatic improvement in result quality! ';
      } else if (improvementPercent > 5) {
        description += 'Noticeable improvement in results. ';
      } else if (improvementPercent > 0) {
        description += 'Slight improvement detected. ';
      } else if (improvementPercent < -10) {
        description += 'Results quality has decreased significantly. ';
      } else if (improvementPercent < 0) {
        description += 'Slight decrease in result quality. ';
      }
    }
    
    return description;
  }, [businessMetrics]);
  
  // Render financial insights
  const renderFinancialInsights = () => {
    const { financialInsights } = businessMetrics;
    
    if (!financialInsights || financialInsights.length === 0) {
      return <div className="no-insights">No financial insights detected</div>;
    }
    
    return (
      <div className="financial-insights">
        <h4>Key Financial Insights</h4>
        <ul>
          {financialInsights.map((insight, i) => (
            <li key={i} className="insight-item">
              <div className="insight-text">{insight.text}</div>
              <div className="insight-confidence">
                {Math.round(insight.similarity * 100)}% confidence
              </div>
            </li>
          ))}
        </ul>
      </div>
    );
  };
  
  // Render category coverage
  const renderCategoryCoverage = () => {
    const { categoryCoverage } = businessMetrics;
    
    if (!categoryCoverage || Object.keys(categoryCoverage).length === 0) {
      return null;
    }
    
    return (
      <div className="category-coverage">
        <h4>Financial Category Coverage</h4>
        <div className="category-bars">
          {Object.entries(categoryCoverage).map(([category, coverage]) => (
            <div key={category} className="category-bar-item">
              <div className="category-name">{category}</div>
              <div className="category-bar-container">
                <div 
                  className="category-bar-fill"
                  style={{ width: `${Math.round(coverage * 100)}%` }}
                ></div>
              </div>
              <div className="category-percentage">{Math.round(coverage * 100)}%</div>
            </div>
          ))}
        </div>
      </div>
    );
  };
  
  // Render comparison controls
  const renderComparisonControls = () => {
    if (!comparisonEnabled || !comparisonPoints) {
      return null;
    }
    
    return (
      <div className="comparison-controls">
        <div className="view-buttons">
          <button 
            className={`view-button ${activeView === 'current' ? 'active' : ''}`}
            onClick={() => setActiveView('current')}
          >
            {baselineLabel}
          </button>
          <button 
            className={`view-button ${activeView === 'comparison' ? 'active' : ''}`}
            onClick={() => setActiveView('comparison')}
          >
            {comparisonLabel}
          </button>
          <button 
            className={`view-button ${activeView === 'split' ? 'active' : ''}`}
            onClick={() => setActiveView('split')}
          >
            Side by Side
          </button>
        </div>
        
        {activeView === 'split' ? (
          <div className="comparison-metrics">
            <div className="metric-improvement">
              <span className="metric-label">Improvement:</span>
              <span className={`metric-value ${businessMetrics.improvementPercent > 0 ? 'positive' : businessMetrics.improvementPercent < 0 ? 'negative' : ''}`}>
                {businessMetrics.improvementPercent > 0 ? '+' : ''}
                {businessMetrics.improvementPercent.toFixed(1)}%
              </span>
            </div>
          </div>
        ) : null}
      </div>
    );
  };
  
  // Render share controls
  const renderShareControls = () => {
    if (!enableSharing) {
      return null;
    }
    
    return (
      <div className="share-controls">
        {!shareUrl ? (
          <button 
            className="share-button"
            onClick={generateShareableReport}
            disabled={isGeneratingReport}
          >
            {isGeneratingReport ? 'Generating...' : 'Create Shareable Report'}
          </button>
        ) : (
          <div className="share-url-container">
            <input 
              type="text" 
              className="share-url" 
              value={shareUrl} 
              readOnly 
              onClick={(e) => e.target.select()}
            />
            <button 
              className="copy-button"
              onClick={() => {
                navigator.clipboard.writeText(shareUrl);
                alert('Share URL copied to clipboard!');
              }}
            >
              Copy
            </button>
            <button 
              className="close-button"
              onClick={() => setShareUrl('')}
            >
              ×
            </button>
          </div>
        )}
      </div>
    );
  };
  
  // Main render function
  return (
    <div className={`live-embedding-visualizer ${className}`} style={style}>
      {/* Business metrics section */}
      {showBusinessMetrics && (
        <div className="business-metrics-container">
          <div className="business-description">
            {getBusinessDescription()}
          </div>
          
          <div className="metrics-dashboard">
            <div className="metric-card confidence">
              <div className="metric-title">Confidence</div>
              <div className="metric-value">{(businessMetrics.confidenceScore * 100).toFixed(0)}%</div>
              <div className="metric-gauge">
                <div 
                  className="metric-gauge-fill"
                  style={{ 
                    width: `${businessMetrics.confidenceScore * 100}%`,
                    backgroundColor: businessMetrics.confidenceScore >= 0.7 ? '#4caf50' : 
                                    businessMetrics.confidenceScore >= 0.4 ? '#ff9800' : '#f44336'
                  }}
                ></div>
              </div>
            </div>
            
            <div className="metric-card relevance">
              <div className="metric-title">Relevance</div>
              <div className="metric-value">{(businessMetrics.relevanceScore * 100).toFixed(0)}%</div>
              <div className="metric-gauge">
                <div 
                  className="metric-gauge-fill"
                  style={{ 
                    width: `${businessMetrics.relevanceScore * 100}%`,
                    backgroundColor: businessMetrics.relevanceScore >= 0.7 ? '#4caf50' : 
                                    businessMetrics.relevanceScore >= 0.4 ? '#ff9800' : '#f44336'
                  }}
                ></div>
              </div>
            </div>
            
            {Math.abs(businessMetrics.improvementPercent) > 1 && (
              <div className="metric-card improvement">
                <div className="metric-title">Improvement</div>
                <div className={`metric-value ${businessMetrics.improvementPercent > 0 ? 'positive' : businessMetrics.improvementPercent < 0 ? 'negative' : ''}`}>
                  {businessMetrics.improvementPercent > 0 ? '+' : ''}
                  {businessMetrics.improvementPercent.toFixed(1)}%
                </div>
              </div>
            )}
          </div>
          
          {/* Financial insights */}
          {businessContext === 'financial' && renderFinancialInsights()}
          
          {/* Category coverage */}
          {financialCategories && showFinancialLegend && renderCategoryCoverage()}
        </div>
      )}
      
      {/* Comparison controls */}
      {renderComparisonControls()}
      
      {/* Visualization section */}
      <div className={`visualizer-container ${activeView === 'split' ? 'split-view' : ''}`}>
        {/* Main visualizer */}
        <div className={`visualizer-wrapper ${activeView === 'comparison' ? 'hidden' : ''}`}>
          <div className={`loading-indicator ${isAnimating ? 'visible' : ''}`}>
            <div className="spinner"></div>
            <div className="update-text">Updating embeddings...</div>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${transitionProgress * 100}%` }}></div>
            </div>
          </div>
          
          <EmbeddingVisualizerReact
            points={currentPoints}
            metadata={currentMetadata}
            similarities={currentSimilarities}
            queryPoint={currentQueryPoint}
            clusters={currentClusters}
            width={activeView === 'split' ? '100%' : width}
            height={height}
            method={method}
            colorScheme={colorScheme}
            showLabels={showLabels}
            showAxes={showAxes}
            showGrid={showGrid}
            colorblindMode={colorblindMode}
            autoRotate={autoRotate}
            onPointSelected={onPointSelected}
            ref={visualizerRef}
          />
          
          {activeView === 'split' ? (
            <div className="view-label baseline-label">{baselineLabel}</div>
          ) : null}
        </div>
        
        {/* Comparison visualizer (when in split or comparison view) */}
        {(activeView === 'split' || activeView === 'comparison') && comparisonPoints && (
          <div className="visualizer-wrapper">
            <EmbeddingVisualizerReact
              points={comparisonPoints}
              metadata={currentMetadata}
              similarities={currentSimilarities}
              queryPoint={currentQueryPoint}
              clusters={currentClusters}
              width="100%"
              height={height}
              method={method}
              colorScheme={colorScheme}
              showLabels={showLabels}
              showAxes={showAxes}
              showGrid={showGrid}
              colorblindMode={colorblindMode}
              autoRotate={autoRotate}
              ref={comparisonVisualizerRef}
            />
            
            <div className="view-label comparison-label">{comparisonLabel}</div>
          </div>
        )}
      </div>
      
      {/* Share controls */}
      {renderShareControls()}
      
      <style jsx>{`
        .live-embedding-visualizer {
          display: flex;
          flex-direction: column;
          width: 100%;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
        }
        
        .business-metrics-container {
          margin-bottom: 1rem;
          padding: 1rem;
          background-color: #f5f5f5;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .business-description {
          font-size: 1rem;
          line-height: 1.5;
          margin-bottom: 1rem;
          color: #333;
        }
        
        .metrics-dashboard {
          display: flex;
          flex-wrap: wrap;
          gap: 1rem;
          margin-bottom: 1rem;
        }
        
        .metric-card {
          flex: 1;
          min-width: 150px;
          padding: 1rem;
          border-radius: 8px;
          background-color: white;
          box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .metric-title {
          font-size: 0.9rem;
          color: #666;
          margin-bottom: 0.5rem;
        }
        
        .metric-value {
          font-size: 1.5rem;
          font-weight: bold;
          color: #333;
          margin-bottom: 0.5rem;
        }
        
        .metric-value.positive {
          color: #4caf50;
        }
        
        .metric-value.negative {
          color: #f44336;
        }
        
        .metric-gauge {
          height: 6px;
          background-color: #eee;
          border-radius: 3px;
          overflow: hidden;
        }
        
        .metric-gauge-fill {
          height: 100%;
          background-color: #4caf50;
          transition: width 0.5s ease-out;
        }
        
        .financial-insights {
          margin-top: 1rem;
        }
        
        .financial-insights h4 {
          font-size: 1rem;
          margin: 0 0 0.5rem 0;
          color: #333;
        }
        
        .financial-insights ul {
          list-style: none;
          padding: 0;
          margin: 0;
        }
        
        .insight-item {
          display: flex;
          justify-content: space-between;
          padding: 0.5rem;
          margin-bottom: 0.5rem;
          background-color: white;
          border-radius: 4px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        .insight-text {
          flex: 1;
          font-size: 0.9rem;
        }
        
        .insight-confidence {
          font-size: 0.9rem;
          font-weight: bold;
          color: #4caf50;
        }
        
        .category-coverage {
          margin-top: 1rem;
        }
        
        .category-coverage h4 {
          font-size: 1rem;
          margin: 0 0 0.5rem 0;
          color: #333;
        }
        
        .category-bars {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }
        
        .category-bar-item {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }
        
        .category-name {
          width: 100px;
          font-size: 0.9rem;
          color: #666;
        }
        
        .category-bar-container {
          flex: 1;
          height: 12px;
          background-color: #eee;
          border-radius: 6px;
          overflow: hidden;
        }
        
        .category-bar-fill {
          height: 100%;
          background-color: #2196f3;
          transition: width 0.5s ease-out;
        }
        
        .category-percentage {
          width: 40px;
          font-size: 0.9rem;
          color: #666;
          text-align: right;
        }
        
        .comparison-controls {
          margin-bottom: 1rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        
        .view-buttons {
          display: flex;
          gap: 0.5rem;
        }
        
        .view-button {
          padding: 0.5rem 1rem;
          border: 1px solid #ddd;
          border-radius: 4px;
          background-color: white;
          font-size: 0.9rem;
          cursor: pointer;
          transition: all 0.2s ease;
        }
        
        .view-button.active {
          background-color: #2196f3;
          color: white;
          border-color: #2196f3;
        }
        
        .comparison-metrics {
          display: flex;
          gap: 1rem;
        }
        
        .metric-improvement {
          font-size: 0.9rem;
        }
        
        .metric-label {
          color: #666;
          margin-right: 0.5rem;
        }
        
        .visualizer-container {
          position: relative;
          width: 100%;
        }
        
        .visualizer-container.split-view {
          display: flex;
          gap: 1rem;
        }
        
        .visualizer-wrapper {
          position: relative;
          flex: 1;
        }
        
        .visualizer-wrapper.hidden {
          display: none;
        }
        
        .view-label {
          position: absolute;
          top: 10px;
          left: 10px;
          padding: 4px 8px;
          background-color: rgba(0,0,0,0.6);
          color: white;
          font-size: 0.8rem;
          border-radius: 4px;
        }
        
        .loading-indicator {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
          background-color: rgba(255,255,255,0.7);
          z-index: 10;
          opacity: 0;
          pointer-events: none;
          transition: opacity 0.3s ease;
        }
        
        .loading-indicator.visible {
          opacity: 1;
        }
        
        .spinner {
          width: 40px;
          height: 40px;
          border: 3px solid rgba(0,0,0,0.1);
          border-top: 3px solid #2196f3;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin-bottom: 1rem;
        }
        
        .update-text {
          font-size: 1rem;
          color: #333;
          margin-bottom: 1rem;
        }
        
        .progress-bar {
          width: 200px;
          height: 6px;
          background-color: #eee;
          border-radius: 3px;
          overflow: hidden;
        }
        
        .progress-fill {
          height: 100%;
          background-color: #2196f3;
          transition: width 0.2s linear;
        }
        
        .share-controls {
          margin-top: 1rem;
          display: flex;
          justify-content: flex-end;
        }
        
        .share-button {
          padding: 0.5rem 1rem;
          background-color: #2196f3;
          color: white;
          border: none;
          border-radius: 4px;
          font-size: 0.9rem;
          cursor: pointer;
          transition: background-color 0.2s ease;
        }
        
        .share-button:hover {
          background-color: #1976d2;
        }
        
        .share-button:disabled {
          background-color: #bdbdbd;
          cursor: not-allowed;
        }
        
        .share-url-container {
          display: flex;
          width: 100%;
        }
        
        .share-url {
          flex: 1;
          padding: 0.5rem;
          border: 1px solid #ddd;
          border-radius: 4px 0 0 4px;
          font-size: 0.9rem;
        }
        
        .copy-button, .close-button {
          padding: 0.5rem 1rem;
          border: none;
          font-size: 0.9rem;
          cursor: pointer;
        }
        
        .copy-button {
          background-color: #4caf50;
          color: white;
        }
        
        .close-button {
          background-color: #f44336;
          color: white;
          border-radius: 0 4px 4px 0;
        }
        
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
          .metrics-dashboard {
            flex-direction: column;
          }
          
          .visualizer-container.split-view {
            flex-direction: column;
          }
          
          .comparison-controls {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.5rem;
          }
        }
      `}</style>
    </div>
  );
};

export default LiveEmbeddingVisualizer;