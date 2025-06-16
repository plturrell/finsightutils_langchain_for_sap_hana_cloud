/**
 * BusinessMetricsTranslator.js
 * 
 * Utility to translate technical embedding metrics into business-meaningful 
 * descriptions and insights, with a focus on financial domain applications.
 */

/**
 * Financial terminology dictionary for enhanced business descriptions
 */
const FINANCIAL_TERMS = {
  // Financial statement terms
  revenue: "total income before expenses",
  earnings: "profit after expenses",
  ebitda: "earnings before interest, taxes, depreciation and amortization",
  roi: "return on investment",
  cagr: "compound annual growth rate",
  eps: "earnings per share",
  pe_ratio: "price to earnings ratio",
  
  // Financial events
  dividend: "payment to shareholders",
  stock_split: "division of existing shares into multiple shares",
  merger: "combination of two companies",
  acquisition: "purchase of one company by another",
  
  // Financial trends
  bull_market: "market condition with rising prices",
  bear_market: "market condition with falling prices",
  volatility: "price fluctuation measurement",
  liquidity: "ability to convert assets to cash",
  
  // Regulatory terms
  sec_filing: "documentation submitted to Securities and Exchange Commission",
  compliance: "adherence to laws and regulations",
  audit: "formal examination of financial records",
  gaap: "Generally Accepted Accounting Principles"
};

/**
 * Business context categories for tailoring metric descriptions
 */
const BUSINESS_CONTEXTS = [
  'financial', 
  'investor_relations', 
  'regulatory', 
  'risk_management',
  'market_research',
  'performance_analysis'
];

/**
 * Translate similarity scores into business confidence language
 * @param {number} similarityScore - Raw similarity score (0-1)
 * @param {string} context - Business context for language customization
 * @returns {object} Confidence description with text and level
 */
function translateSimilarityToConfidence(similarityScore, context = 'financial') {
  // Confidence thresholds
  const thresholds = {
    financial: { high: 0.85, medium: 0.65, low: 0.4 },
    investor_relations: { high: 0.9, medium: 0.75, low: 0.5 },
    regulatory: { high: 0.92, medium: 0.8, low: 0.6 },
    risk_management: { high: 0.9, medium: 0.75, low: 0.5 },
    market_research: { high: 0.8, medium: 0.6, low: 0.4 },
    performance_analysis: { high: 0.85, medium: 0.7, low: 0.5 }
  };
  
  // Default to financial thresholds if context not found
  const { high, medium, low } = thresholds[context] || thresholds.financial;
  
  // Confidence level determination
  let level, text;
  
  if (similarityScore >= high) {
    level = 'high';
    text = context === 'regulatory' 
      ? 'Highly compliant match with very strong correlation'
      : 'Very high confidence match with strong relevance';
  } else if (similarityScore >= medium) {
    level = 'medium';
    text = context === 'regulatory'
      ? 'Moderately compliant match with good correlation'
      : 'Good confidence in results with meaningful relevance';
  } else if (similarityScore >= low) {
    level = 'low';
    text = context === 'regulatory'
      ? 'Minimally compliant match with some correlation'
      : 'Limited confidence in results with some relevance';
  } else {
    level = 'very_low';
    text = context === 'regulatory'
      ? 'Non-compliant match with minimal correlation'
      : 'Very low confidence in results with questionable relevance';
  }
  
  // Add percentage for clarity
  const percentage = Math.round(similarityScore * 100);
  
  return {
    level,
    text,
    percentage,
    description: `${text} (${percentage}% match)`
  };
}

/**
 * Convert technical metrics into business impact descriptions
 * @param {object} metrics - Technical metrics from embedding results
 * @param {string} context - Business context for language customization
 * @returns {object} Business-oriented descriptions and insights
 */
function translateMetricsToBusinessValue(metrics, context = 'financial') {
  const {
    averageSimilarity = 0,
    topSimilarity = 0, 
    resultCount = 0,
    relevantCount = 0,
    categories = {},
    processingTime = 0,
    previousAverageSimilarity = null
  } = metrics;
  
  // Calculate derived metrics
  const relevanceRatio = resultCount > 0 ? relevantCount / resultCount : 0;
  const improvementPercent = previousAverageSimilarity !== null
    ? ((averageSimilarity - previousAverageSimilarity) / previousAverageSimilarity) * 100
    : null;
  
  // Create confidence assessment
  const confidenceAssessment = translateSimilarityToConfidence(topSimilarity, context);
  
  // Generate business language for relevance
  let relevanceDescription;
  if (relevanceRatio >= 0.8) {
    relevanceDescription = "Nearly all results are directly relevant to your query";
  } else if (relevanceRatio >= 0.6) {
    relevanceDescription = "Most results are relevant to your query";
  } else if (relevanceRatio >= 0.4) {
    relevanceDescription = "About half of the results are relevant to your query";
  } else if (relevanceRatio >= 0.2) {
    relevanceDescription = "Some results are relevant, but many may not apply";
  } else {
    relevanceDescription = "Few results appear directly relevant to your query";
  }
  
  // Create improvement language if available
  let improvementDescription = null;
  if (improvementPercent !== null) {
    if (improvementPercent >= 20) {
      improvementDescription = `Dramatic improvement of ${improvementPercent.toFixed(1)}% in result quality`;
    } else if (improvementPercent >= 5) {
      improvementDescription = `Noticeable improvement of ${improvementPercent.toFixed(1)}% in result quality`;
    } else if (improvementPercent >= 1) {
      improvementDescription = `Slight improvement of ${improvementPercent.toFixed(1)}% in result quality`;
    } else if (improvementPercent <= -10) {
      improvementDescription = `Significant decrease of ${Math.abs(improvementPercent).toFixed(1)}% in result quality`;
    } else if (improvementPercent < 0) {
      improvementDescription = `Slight decrease of ${Math.abs(improvementPercent).toFixed(1)}% in result quality`;
    } else {
      improvementDescription = "No significant change in result quality";
    }
  }
  
  // Create category coverage insights
  const categoryInsights = Object.entries(categories)
    .map(([category, count]) => {
      const percentage = resultCount > 0 ? (count / resultCount) * 100 : 0;
      return {
        category,
        count,
        percentage,
        description: `${Math.round(percentage)}% ${category} content`
      };
    })
    .sort((a, b) => b.percentage - a.percentage);
  
  // Performance assessment in business terms
  let performanceDescription;
  if (processingTime < 500) {
    performanceDescription = "Results delivered with instant response time";
  } else if (processingTime < 1000) {
    performanceDescription = "Results delivered with fast response time";
  } else if (processingTime < 3000) {
    performanceDescription = "Results delivered with good response time";
  } else {
    performanceDescription = "Results delivered with extended processing time";
  }
  
  // Generate executive summary based on context
  let executiveSummary;
  if (context === 'financial') {
    executiveSummary = `Financial analysis with ${confidenceAssessment.level === 'high' ? 'high' : 'moderate'} confidence. ${relevanceDescription}.`;
    if (improvementDescription) {
      executiveSummary += ` ${improvementDescription}.`;
    }
  } else if (context === 'investor_relations') {
    executiveSummary = `Investor relations data with ${confidenceAssessment.percentage}% confidence match. ${relevanceDescription}.`;
  } else if (context === 'regulatory') {
    executiveSummary = `Regulatory assessment with ${confidenceAssessment.percentage}% compliance correlation. ${relevanceDescription}.`;
  } else {
    executiveSummary = `Analysis with ${confidenceAssessment.percentage}% confidence. ${relevanceDescription}.`;
  }
  
  return {
    confidenceAssessment,
    relevanceDescription,
    improvementDescription,
    categoryInsights,
    performanceDescription,
    executiveSummary,
    
    // Numerical metrics for visualization
    metrics: {
      confidenceScore: topSimilarity,
      relevanceScore: relevanceRatio,
      improvementPercent: improvementPercent || 0,
      categoryDistribution: categoryInsights.reduce((obj, item) => {
        obj[item.category] = item.percentage / 100;
        return obj;
      }, {})
    }
  };
}

/**
 * Translate a technical term to business-friendly language
 * @param {string} term - Technical term to translate
 * @returns {string} Business-friendly description
 */
function translateTechnicalTerm(term) {
  const normalized = term.toLowerCase().trim();
  
  // Check direct match in dictionary
  if (FINANCIAL_TERMS[normalized]) {
    return FINANCIAL_TERMS[normalized];
  }
  
  // Handle compound terms
  const parts = normalized.split('_');
  if (parts.length > 1) {
    const translatedParts = parts.map(part => FINANCIAL_TERMS[part] || part);
    return translatedParts.join(' ');
  }
  
  // Handle embedding-specific terms
  if (normalized === 'similarity') return 'relevance match';
  if (normalized === 'vector') return 'financial data point';
  if (normalized === 'embedding') return 'financial pattern';
  if (normalized === 'cluster') return 'financial category';
  if (normalized === 'dimension') return 'financial attribute';
  if (normalized === 'cosine') return 'pattern match';
  
  // Return original if no translation
  return term;
}

/**
 * Generate natural language insights from embedding results
 * @param {object} embeddingResults - Technical embedding results
 * @param {Array} documents - Source documents
 * @param {string} context - Business context
 * @returns {Array} Natural language insights about the data
 */
function generateBusinessInsights(embeddingResults, documents, context = 'financial') {
  if (!embeddingResults || !documents || documents.length === 0) {
    return [];
  }
  
  const insights = [];
  
  // Identify patterns in top results
  const topDocuments = documents
    .map((doc, i) => ({ doc, similarity: embeddingResults.similarities?.[i] || 0 }))
    .filter(item => item.similarity > 0.6)
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, 5);
  
  if (topDocuments.length > 0) {
    // Analyze document dates if available
    const datesPresent = topDocuments.filter(item => item.doc.date).length > 0;
    if (datesPresent) {
      const dates = topDocuments
        .filter(item => item.doc.date)
        .map(item => new Date(item.doc.date));
      
      if (dates.length > 1) {
        // Sort dates
        dates.sort((a, b) => a - b);
        
        // Calculate date range
        const oldestDate = dates[0];
        const newestDate = dates[dates.length - 1];
        const dateRangeMonths = (newestDate - oldestDate) / (1000 * 60 * 60 * 24 * 30);
        
        if (dateRangeMonths > 12) {
          insights.push(`Results span ${Math.round(dateRangeMonths / 12)} years, indicating long-term trend analysis`);
        } else if (dateRangeMonths > 1) {
          insights.push(`Results span ${Math.round(dateRangeMonths)} months, indicating medium-term pattern analysis`);
        } else {
          insights.push(`Results focus on recent data (past month), indicating current situation analysis`);
        }
      }
    }
    
    // Analyze common financial terms
    const commonTerms = {};
    const financialTermsRegex = new RegExp(
      Object.keys(FINANCIAL_TERMS).join('|'), 'gi'
    );
    
    topDocuments.forEach(item => {
      const content = item.doc.title + ' ' + (item.doc.content || '');
      const matches = content.match(financialTermsRegex) || [];
      
      matches.forEach(match => {
        const term = match.toLowerCase();
        commonTerms[term] = (commonTerms[term] || 0) + 1;
      });
    });
    
    // Get top mentioned terms
    const topTerms = Object.entries(commonTerms)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3);
    
    if (topTerms.length > 0) {
      insights.push(`Key financial concepts: ${topTerms.map(([term]) => term).join(', ')}`);
    }
    
    // Analyze sentiment if available
    if (topDocuments.some(item => item.doc.sentiment)) {
      const sentiments = topDocuments
        .filter(item => item.doc.sentiment)
        .map(item => item.doc.sentiment);
      
      const positiveCount = sentiments.filter(s => s > 0).length;
      const negativeCount = sentiments.filter(s => s < 0).length;
      const neutralCount = sentiments.length - positiveCount - negativeCount;
      
      if (positiveCount > negativeCount && positiveCount > neutralCount) {
        insights.push(`Predominantly positive financial outlook (${Math.round(positiveCount/sentiments.length*100)}% positive)`);
      } else if (negativeCount > positiveCount && negativeCount > neutralCount) {
        insights.push(`Predominantly cautious financial outlook (${Math.round(negativeCount/sentiments.length*100)}% cautious)`);
      } else {
        insights.push(`Balanced or neutral financial perspective`);
      }
    }
  }
  
  // Add domain-specific insights based on context
  if (context === 'financial') {
    insights.push(`Financial data analysis optimized for financial statements and reports`);
  } else if (context === 'investor_relations') {
    insights.push(`Results optimized for investor communications and shareholder relevance`);
  } else if (context === 'regulatory') {
    insights.push(`Analysis focused on regulatory compliance and documentation`);
  } else if (context === 'risk_management') {
    insights.push(`Results prioritized for risk assessment and mitigation insights`);
  }
  
  return insights;
}

// Export the utility functions
export {
  translateSimilarityToConfidence,
  translateMetricsToBusinessValue,
  translateTechnicalTerm,
  generateBusinessInsights,
  FINANCIAL_TERMS,
  BUSINESS_CONTEXTS
};