"""
Financial domain calibration datasets for TensorRT INT8 quantization.

This module provides specialized calibration datasets for financial domain models,
ensuring optimal INT8 quantization for financial text embedding models.
"""

import os
import logging
import json
import random
from typing import List, Dict, Any, Optional, Set, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# Financial terminology by category
FINANCIAL_TERMS = {
    "accounting": [
        "assets", "liabilities", "equity", "revenue", "expense", "profit", "loss", 
        "balance sheet", "income statement", "cash flow statement", "journal entry",
        "accounts receivable", "accounts payable", "accrual", "depreciation", "amortization",
        "GAAP", "IFRS", "EBITDA", "EBIT", "gross margin", "operating margin", "net margin",
        "working capital", "retained earnings", "goodwill", "intangible assets", "tangible assets",
        "impairment", "audit", "fiscal year", "quarter", "10-K", "10-Q", "annual report",
        "book value", "fair value", "historical cost", "inventory", "COGS", "FIFO", "LIFO",
        "weighted average", "tax expense", "deferred tax", "effective tax rate", "capitalization",
        "operating expenses", "non-operating expenses", "extraordinary items", "discontinued operations"
    ],
    "investment": [
        "stock", "bond", "mutual fund", "ETF", "index fund", "hedge fund", "private equity",
        "venture capital", "angel investor", "portfolio", "diversification", "asset allocation",
        "risk tolerance", "ROI", "return on investment", "capital gain", "dividend", "yield",
        "P/E ratio", "price-to-earnings", "EPS", "earnings per share", "market cap", "beta",
        "alpha", "volatility", "standard deviation", "sharpe ratio", "modern portfolio theory",
        "efficient frontier", "CAPM", "arbitrage", "fundamental analysis", "technical analysis",
        "value investing", "growth investing", "income investing", "day trading", "swing trading",
        "position trading", "dollar cost averaging", "IPO", "initial public offering", "secondary offering",
        "underwriter", "prospectus", "due diligence", "bull market", "bear market", "correction",
        "recession", "depression", "inflation", "deflation", "stagflation", "interest rate",
        "federal funds rate", "yield curve", "inverted yield curve", "treasury", "junk bond",
        "investment grade", "credit rating", "S&P", "Moody's", "Fitch", "preferred stock",
        "common stock", "derivative", "option", "futures", "forward", "swap", "call option",
        "put option", "strike price", "expiration date", "margin", "leverage", "short selling",
        "long position", "short position", "market order", "limit order", "stop loss order",
        "trailing stop", "bid", "ask", "spread", "liquidity", "volume", "market maker",
        "broker", "dealer", "fiduciary", "fee-only advisor", "asset under management", "AUM",
        "load fund", "no-load fund", "expense ratio", "management fee", "performance fee"
    ],
    "banking": [
        "deposit", "withdrawal", "transfer", "balance", "checking account", "savings account",
        "money market account", "certificate of deposit", "CD", "interest rate", "APY",
        "annual percentage yield", "APR", "annual percentage rate", "compound interest",
        "simple interest", "principal", "term", "maturity", "early withdrawal penalty",
        "overdraft", "insufficient funds", "NSF fee", "minimum balance", "direct deposit",
        "wire transfer", "ACH transfer", "routing number", "account number", "online banking",
        "mobile banking", "ATM", "debit card", "credit card", "secured card", "unsecured card",
        "credit limit", "available credit", "credit utilization", "balance transfer", "cash advance",
        "grace period", "late fee", "over-limit fee", "foreign transaction fee", "annual fee",
        "rewards program", "cash back", "points", "miles", "statement", "billing cycle",
        "due date", "minimum payment", "credit score", "FICO score", "credit report", "credit bureau",
        "Equifax", "Experian", "TransUnion", "credit history", "credit inquiry", "hard inquiry",
        "soft inquiry", "secured loan", "unsecured loan", "collateral", "personal loan", "auto loan",
        "mortgage", "home equity loan", "HELOC", "line of credit", "origination fee", "closing costs",
        "escrow", "amortization schedule", "fixed rate", "variable rate", "adjustable rate",
        "ARM", "refinance", "prepayment penalty", "points", "discount points", "loan-to-value ratio",
        "LTV", "debt-to-income ratio", "DTI", "pre-approval", "pre-qualification", "underwriting",
        "foreclosure", "short sale", "deed in lieu", "default", "delinquency", "collection",
        "charge-off", "bankruptcy", "Chapter 7", "Chapter 13", "debt settlement", "debt consolidation",
        "debt management", "credit counseling", "FDIC", "NCUA", "prime rate", "LIBOR", "SOFR"
    ],
    "market_analysis": [
        "trend", "pattern", "indicator", "oscillator", "moving average", "simple moving average",
        "exponential moving average", "MACD", "RSI", "relative strength index", "stochastic oscillator",
        "Bollinger bands", "support", "resistance", "breakout", "breakdown", "consolidation",
        "divergence", "convergence", "overbought", "oversold", "momentum", "volume", "open interest",
        "gap", "head and shoulders", "double top", "double bottom", "triangle", "wedge", "flag",
        "pennant", "channel", "Fibonacci retracement", "Elliott wave", "Dow theory", "candlestick",
        "doji", "hammer", "hanging man", "engulfing pattern", "morning star", "evening star",
        "technical indicator", "fundamental indicator", "economic indicator", "leading indicator",
        "lagging indicator", "coincident indicator", "market breadth", "advance-decline line",
        "new highs-new lows", "TICK", "VIX", "volatility index", "put-call ratio", "sentiment",
        "bullish", "bearish", "contrarian", "short interest", "days to cover", "institutional ownership",
        "insider trading", "insider buying", "insider selling", "sector rotation", "market cycle",
        "business cycle", "expansion", "peak", "contraction", "trough", "recovery", "seasonality",
        "January effect", "sell in May and go away", "Santa Claus rally", "window dressing",
        "quarterly earnings", "earnings season", "guidance", "surprise", "beat", "miss", "revision",
        "upgrade", "downgrade", "buy rating", "sell rating", "hold rating", "overweight", "underweight",
        "market weight", "outperform", "underperform", "market perform", "price target", "consensus estimate"
    ],
    "economics": [
        "microeconomics", "macroeconomics", "supply", "demand", "equilibrium", "elasticity",
        "inelastic", "elastic", "substitute", "complement", "normal good", "inferior good",
        "luxury good", "necessity good", "utility", "marginal utility", "diminishing returns",
        "opportunity cost", "comparative advantage", "absolute advantage", "specialization",
        "division of labor", "production possibility frontier", "economies of scale", "diseconomies of scale",
        "perfect competition", "monopoly", "oligopoly", "monopolistic competition", "barriers to entry",
        "market failure", "externality", "public good", "private good", "free rider problem",
        "tragedy of the commons", "GDP", "gross domestic product", "GNP", "gross national product",
        "real GDP", "nominal GDP", "GDP per capita", "economic growth", "business cycle",
        "expansion", "recession", "depression", "recovery", "unemployment rate", "natural rate of unemployment",
        "frictional unemployment", "structural unemployment", "cyclical unemployment", "inflation",
        "deflation", "hyperinflation", "disinflation", "stagflation", "CPI", "consumer price index",
        "PPI", "producer price index", "cost-push inflation", "demand-pull inflation", "monetary policy",
        "fiscal policy", "expansionary policy", "contractionary policy", "federal reserve", "central bank",
        "interest rate", "discount rate", "federal funds rate", "reserve requirement", "open market operations",
        "quantitative easing", "tightening", "money supply", "M0", "M1", "M2", "velocity of money",
        "liquidity trap", "IS-LM model", "aggregate demand", "aggregate supply", "Phillips curve",
        "Laffer curve", "multiplier effect", "crowding out", "national debt", "budget deficit",
        "budget surplus", "balanced budget", "automatic stabilizer", "discretionary policy",
        "progressive tax", "regressive tax", "proportional tax", "tax incidence", "deadweight loss",
        "economic indicator", "leading indicator", "lagging indicator", "coincident indicator",
        "exchange rate", "fixed exchange rate", "floating exchange rate", "appreciation", "depreciation",
        "devaluation", "revaluation", "currency intervention", "purchasing power parity", "PPP",
        "balance of payments", "current account", "capital account", "trade deficit", "trade surplus",
        "protectionism", "tariff", "quota", "subsidy", "dumping", "comparative advantage",
        "absolute advantage", "terms of trade", "free trade", "trade agreement", "WTO", "IMF", "World Bank"
    ],
    "corporate_finance": [
        "capital structure", "debt financing", "equity financing", "leverage", "debt-to-equity ratio",
        "weighted average cost of capital", "WACC", "cost of debt", "cost of equity", "capital asset pricing model",
        "CAPM", "beta", "risk-free rate", "market risk premium", "dividend discount model", "Gordon growth model",
        "discounted cash flow", "DCF", "free cash flow", "FCFF", "FCFE", "terminal value", "perpetuity",
        "growth rate", "hurdle rate", "internal rate of return", "IRR", "net present value", "NPV",
        "payback period", "profitability index", "return on investment", "ROI", "return on assets", "ROA",
        "return on equity", "ROE", "return on capital employed", "ROCE", "economic value added", "EVA",
        "market value added", "MVA", "enterprise value", "EV", "EV/EBITDA", "EV/Sales", "P/E ratio",
        "P/B ratio", "P/S ratio", "PEG ratio", "dividend yield", "dividend payout ratio", "retention ratio",
        "sustainable growth rate", "capital expenditure", "CAPEX", "operating expenditure", "OPEX",
        "working capital", "current ratio", "quick ratio", "cash ratio", "days sales outstanding", "DSO",
        "days inventory outstanding", "DIO", "days payable outstanding", "DPO", "cash conversion cycle",
        "operating cycle", "financial leverage", "operating leverage", "combined leverage", "break-even point",
        "contribution margin", "fixed cost", "variable cost", "semi-variable cost", "step cost",
        "relevant cost", "sunk cost", "opportunity cost", "incremental cost", "marginal cost",
        "average cost", "absorption costing", "activity-based costing", "standard costing", "variance analysis",
        "favorable variance", "unfavorable variance", "budget", "master budget", "operating budget",
        "cash budget", "flexible budget", "static budget", "zero-based budgeting", "capital budgeting",
        "treasury", "cash management", "liquidity management", "credit management", "accounts receivable management",
        "inventory management", "merger", "acquisition", "horizontal merger", "vertical merger",
        "conglomerate merger", "synergy", "corporate restructuring", "spin-off", "split-off", "carve-out",
        "divestiture", "leveraged buyout", "LBO", "management buyout", "MBO", "hostile takeover",
        "tender offer", "proxy fight", "golden parachute", "poison pill", "white knight", "greenmail"
    ],
    "fintech": [
        "cryptocurrency", "bitcoin", "ethereum", "altcoin", "token", "initial coin offering", "ICO",
        "security token offering", "STO", "decentralized finance", "DeFi", "blockchain", "distributed ledger",
        "smart contract", "consensus mechanism", "proof of work", "proof of stake", "mining", "hash",
        "wallet", "private key", "public key", "cold storage", "hot wallet", "exchange", "decentralized exchange",
        "DEX", "liquidity pool", "automated market maker", "AMM", "yield farming", "staking", "lending protocol",
        "borrowing protocol", "collateralization ratio", "liquidation", "governance token", "utility token",
        "security token", "non-fungible token", "NFT", "gas fee", "robo-advisor", "algorithmic trading",
        "high-frequency trading", "HFT", "digital banking", "neobank", "challenger bank", "mobile banking",
        "peer-to-peer lending", "P2P lending", "crowdfunding", "equity crowdfunding", "rewards crowdfunding",
        "payment processor", "payment gateway", "digital wallet", "mobile wallet", "contactless payment",
        "NFC payment", "QR code payment", "point of sale", "POS system", "buy now pay later", "BNPL",
        "open banking", "API banking", "banking as a service", "BaaS", "personal finance management", "PFM",
        "insurtech", "on-demand insurance", "usage-based insurance", "parametric insurance", "regtech",
        "compliance technology", "KYC", "know your customer", "AML", "anti-money laundering", "biometric authentication",
        "two-factor authentication", "2FA", "multi-factor authentication", "MFA", "AI in finance", "machine learning",
        "predictive analytics", "natural language processing", "NLP", "chatbot", "virtual assistant",
        "robotic process automation", "RPA", "quantum computing", "cloud computing", "SaaS", "PaaS", "IaaS",
        "big data", "data mining", "data analytics", "data visualization", "API", "sandbox", "webhook",
        "encryption", "cybersecurity", "penetration testing", "vulnerability assessment", "regulatory compliance",
        "GDPR", "PSD2", "MiFID II", "Dodd-Frank", "Basel III", "Sarbanes-Oxley", "SOX"
    ]
}

# Financial document templates
FINANCIAL_DOCUMENT_TEMPLATES = [
    # Quarterly reports
    "For the quarter ended {quarter} {year}, {company} reported {revenue} in revenue, a {growth}% {direction} from the same period last year. EBITDA was {ebitda}, with an EBITDA margin of {margin}%. {sector} segment revenue was {segment_revenue}, representing {segment_percent}% of total revenue. The company reported earnings per share of {eps}, {vs_estimate} analyst estimates of {estimate}. {outlook}",
    
    # Annual reports
    "In fiscal year {year}, {company} achieved total revenue of {revenue}, reflecting a {growth}% {direction} compared to the previous year. Net income was {net_income}, resulting in a net profit margin of {margin}%. Return on equity stood at {roe}%, while return on assets was {roa}%. The board approved a {dividend} dividend per share, representing a {payout}% payout ratio. {guidance}",
    
    # Earnings calls
    "During the earnings call for {quarter} {year}, {company}'s management discussed the {performance} performance in the {sector} segment, which grew by {growth}% year-over-year. CEO {ceo_name} highlighted the strategic initiatives to {initiative} in the coming quarters. When asked about {topic}, CFO {cfo_name} explained that {explanation}. Analysts from {bank} raised concerns about {concern}, to which management responded {response}.",
    
    # Financial news
    "Breaking: {company} announces {announcement} in a move that could {impact} the {sector} industry. Analysts from {bank} view this as {sentiment} for the company's long-term growth prospects. The stock {stock_action} by {percent}% in early trading, with trading volume {volume} times the daily average. Competitors like {competitor} also saw {comp_action} in their share prices on the news.",
    
    # Market analysis
    "Market Analysis: The {index} {index_action} by {percent}% today, led by {sector} stocks amid {reason}. Technical indicators suggest {technical_outlook}, with the index currently trading {ma_relation} its 200-day moving average. Volume was {volume} compared to the 30-day average. Key resistance levels remain at {resistance}, while support is established around {support}. {analyst} from {firm} recommends {recommendation} based on {rationale}.",
    
    # Economic reports
    "The latest {econ_report} shows {econ_metric} {econ_action} to {econ_value} in {month} {year}, {vs_expectation} economists' expectations of {expectation}. This marks the {consecutive} consecutive {direction} and suggests {implication} for the broader economy. {agency} officials noted that {official_comment}. Markets reacted with {market_reaction}, as investors recalibrated their expectations for {fed_action} at the next Federal Reserve meeting.",
    
    # Investment research
    "Investment Research: We initiate coverage of {company} ({ticker}) with a {rating} rating and a price target of {price_target}, implying {upside}% upside potential. Our thesis is based on {thesis_points}. Key risks include {risks}. We forecast revenue growth of {forecast_growth}% CAGR over the next {years} years, driven by {growth_drivers}. Our valuation is based on {valuation_method} with a {multiple}x multiple on {metric}.",
    
    # Financial statements
    "Balance Sheet Analysis: As of {date}, {company} reported total assets of {assets}, with cash and equivalents of {cash}. Total liabilities stood at {liabilities}, including long-term debt of {debt}. The debt-to-equity ratio was {de_ratio}x, {vs_industry} the industry average of {industry_avg}. Working capital was {working_capital}, resulting in a current ratio of {current_ratio}x. Inventory turnover {inventory_turnover} and days sales outstanding {dso} both {vs_previous} compared to the previous quarter.",
    
    # SEC filings
    "In its {filing_type} filing dated {date}, {company} disclosed {disclosure} that could materially impact its financial position. The company also reported {related_party} related party transactions involving {related_entity}. Risk factors highlighted in the filing include {risk_factors}. Management's discussion and analysis focused on {md_and_a}, particularly addressing investor concerns about {concerns}.",
    
    # Financial ratios
    "Financial Ratio Analysis: {company}'s profitability metrics show {metric_trend} trends, with gross margin at {gross_margin}%, operating margin at {operating_margin}%, and net margin at {net_margin}%. Efficiency ratios include asset turnover of {asset_turnover}x and inventory turnover of {inventory_turnover}x. Liquidity metrics remain {liquidity_status}, with a current ratio of {current_ratio}x and quick ratio of {quick_ratio}x. The company's valuation appears {valuation} with a P/E of {pe_ratio}x, EV/EBITDA of {ev_ebitda}x, and P/B of {pb_ratio}x."
]

# Sample values for template variables
TEMPLATE_VALUES = {
    "quarter": ["Q1", "Q2", "Q3", "Q4"],
    "year": list(range(2020, 2026)),
    "company": [
        "Apple", "Microsoft", "Amazon", "Alphabet", "Meta", "Tesla", "JPMorgan Chase",
        "Bank of America", "Goldman Sachs", "Morgan Stanley", "Citigroup", "Wells Fargo",
        "Visa", "Mastercard", "PayPal", "Square", "Robinhood", "Coinbase", "BlackRock",
        "Vanguard", "Fidelity", "Charles Schwab", "UBS", "Credit Suisse", "Deutsche Bank",
        "Barclays", "HSBC", "BNP Paribas", "Santander", "ING Group", "AIG", "Prudential",
        "MetLife", "Chubb", "Allstate", "Progressive", "Berkshire Hathaway", "Walmart",
        "Target", "Home Depot", "Lowe's", "Costco", "Kroger", "Walgreens", "CVS Health",
        "UnitedHealth", "Anthem", "Cigna", "Humana", "Pfizer", "Johnson & Johnson",
        "Merck", "Bristol Myers Squibb", "AbbVie", "Amgen", "Gilead Sciences", "Biogen",
        "Vertex Pharmaceuticals", "Regeneron", "Moderna", "BioNTech", "ExxonMobil",
        "Chevron", "Shell", "BP", "TotalEnergies", "ConocoPhillips", "Occidental Petroleum"
    ],
    "revenue": [
        "$1.2 billion", "$2.5 billion", "$3.7 billion", "$4.3 billion", "$5.8 billion",
        "$6.1 billion", "$7.9 billion", "$8.4 billion", "$9.6 billion", "$10.2 billion",
        "$12.5 billion", "$15.3 billion", "$18.7 billion", "$22.1 billion", "$25.6 billion",
        "$30.2 billion", "$35.8 billion", "$42.3 billion", "$50.1 billion", "$65.9 billion",
        "$72.4 billion", "$85.1 billion", "$94.3 billion", "$108.7 billion", "$123.4 billion"
    ],
    "growth": [str(x) for x in range(1, 36)],
    "direction": ["increase", "decrease", "gain", "decline", "improvement", "reduction"],
    "ebitda": [
        "$450 million", "$780 million", "$1.1 billion", "$1.5 billion", "$1.9 billion",
        "$2.3 billion", "$2.8 billion", "$3.2 billion", "$3.9 billion", "$4.5 billion",
        "$5.1 billion", "$6.3 billion", "$7.2 billion", "$8.5 billion", "$10.1 billion",
        "$12.4 billion", "$15.2 billion", "$18.6 billion", "$22.3 billion", "$26.8 billion"
    ],
    "margin": [str(x) for x in range(8, 46)],
    "sector": [
        "Technology", "Financial Services", "Healthcare", "Consumer Discretionary",
        "Consumer Staples", "Energy", "Utilities", "Materials", "Industrials",
        "Communication Services", "Real Estate", "Cloud Computing", "E-commerce",
        "Digital Payments", "Streaming", "Social Media", "Semiconductor", "Biotech",
        "Pharmaceuticals", "Renewable Energy", "Electric Vehicles", "Artificial Intelligence",
        "Cybersecurity", "Fintech", "Insurtech", "Enterprise Software", "Gaming"
    ],
    "segment_revenue": [
        "$450 million", "$780 million", "$1.1 billion", "$1.5 billion", "$1.9 billion",
        "$2.3 billion", "$2.8 billion", "$3.2 billion", "$3.9 billion", "$4.5 billion"
    ],
    "segment_percent": [str(x) for x in range(10, 71)],
    "eps": [
        "$0.78", "$1.05", "$1.23", "$1.45", "$1.67", "$1.89", "$2.12", "$2.34",
        "$2.56", "$2.78", "$3.01", "$3.25", "$3.48", "$3.72", "$3.95", "$4.23",
        "$4.56", "$4.89", "$5.12", "$5.45", "$5.78", "$6.12", "$6.45", "$6.78"
    ],
    "vs_estimate": [
        "beating", "missing", "in line with", "exceeding", "falling short of",
        "matching", "surpassing", "below", "above", "consistent with"
    ],
    "estimate": [
        "$0.72", "$0.98", "$1.15", "$1.38", "$1.62", "$1.84", "$2.05", "$2.28",
        "$2.52", "$2.72", "$2.95", "$3.18", "$3.42", "$3.65", "$3.88", "$4.15",
        "$4.48", "$4.82", "$5.05", "$5.38", "$5.72", "$6.05", "$6.38", "$6.72"
    ],
    "outlook": [
        "Management raised the full-year guidance, citing strong demand in emerging markets.",
        "The company maintained its conservative outlook due to macroeconomic uncertainties.",
        "Citing supply chain challenges, executives reduced guidance for the upcoming quarter.",
        "The company expects margin pressure to continue due to rising input costs.",
        "Management anticipates accelerating growth in the second half of the year.",
        "The outlook remains cautious as the company navigates regulatory challenges.",
        "Executives expressed confidence in meeting full-year targets despite headwinds.",
        "The company forecasts improving margins as cost-cutting initiatives take effect.",
        "Management expects continued momentum based on the strong product pipeline.",
        "The outlook was revised downward to reflect competitive pressures in key markets."
    ]
}

def generate_calibration_texts(
    count: int = 100,
    min_length: int = 50,
    max_length: int = 500,
    include_categories: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> List[str]:
    """
    Generate financial calibration texts for INT8 quantization.
    
    Parameters
    ----------
    count : int, default=100
        Number of calibration texts to generate
    min_length : int, default=50
        Minimum length of each text in characters
    max_length : int, default=500
        Maximum length of each text in characters
    include_categories : List[str], optional
        Categories to include (all if None)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    List[str]
        List of generated financial texts
    """
    if seed is not None:
        random.seed(seed)
    
    categories = include_categories or list(FINANCIAL_TERMS.keys())
    if not set(categories).issubset(set(FINANCIAL_TERMS.keys())):
        invalid_categories = set(categories) - set(FINANCIAL_TERMS.keys())
        raise ValueError(f"Invalid categories: {invalid_categories}")
    
    # Generate calibration texts
    calibration_texts = []
    
    # Generate from document templates
    template_count = min(count // 2, len(FINANCIAL_DOCUMENT_TEMPLATES))
    for _ in range(template_count):
        template = random.choice(FINANCIAL_DOCUMENT_TEMPLATES)
        
        # Fill in template variables
        text = template
        for var in TEMPLATE_VALUES:
            if "{" + var + "}" in text:
                text = text.replace("{" + var + "}", str(random.choice(TEMPLATE_VALUES[var])))
        
        # Replace any remaining variables with generic values
        import re
        text = re.sub(r'\{[^}]+\}', lambda m: random.choice(["this", "that", "the value", "it"]), text)
        
        calibration_texts.append(text)
    
    # Generate from financial terms
    while len(calibration_texts) < count:
        category = random.choice(categories)
        terms = FINANCIAL_TERMS[category]
        
        # Select a random number of terms
        num_terms = random.randint(5, 15)
        selected_terms = random.sample(terms, min(num_terms, len(terms)))
        
        # Create a coherent sentence or paragraph using these terms
        if random.random() < 0.7:
            # Create a sentence
            connectors = ["and", "or", "as well as", "including", "such as", "like", "especially"]
            sentence_starters = [
                f"The {category} analysis includes ",
                f"Key {category} factors to consider are ",
                f"Important {category} metrics include ",
                f"When evaluating {category}, look at ",
                f"In {category}, we focus on ",
                f"The report highlights these {category} elements: ",
                f"Critical {category} considerations include "
            ]
            
            text = random.choice(sentence_starters)
            for i, term in enumerate(selected_terms):
                if i > 0:
                    if i == len(selected_terms) - 1:
                        text += f" and {term}"
                    else:
                        text += f", {term}"
                else:
                    text += term
            text += "."
        else:
            # Create a paragraph
            sentences = []
            for i in range(random.randint(2, 5)):
                if i == 0:
                    starters = [
                        f"The {category} report analyzes several key metrics.",
                        f"Understanding {category} requires attention to multiple factors.",
                        f"The {category} section examines important indicators.",
                        f"A comprehensive {category} analysis considers various elements.",
                        f"In evaluating {category}, we consider several measurements."
                    ]
                    sentences.append(random.choice(starters))
                
                terms_per_sentence = min(random.randint(1, 4), len(selected_terms) - len(sentences))
                if terms_per_sentence <= 0:
                    break
                
                terms_for_sentence = selected_terms[:terms_per_sentence]
                selected_terms = selected_terms[terms_per_sentence:]
                
                connectors = ["focuses on", "highlights", "emphasizes", "indicates", "shows", "demonstrates", "reveals"]
                sentence = f"The analysis {random.choice(connectors)} {', '.join(terms_for_sentence[:-1])}"
                if len(terms_for_sentence) > 1:
                    sentence += f" and {terms_for_sentence[-1]}"
                else:
                    sentence += terms_for_sentence[-1]
                
                endings = [
                    " as critical indicators.",
                    " for thorough evaluation.",
                    " to guide decision-making.",
                    " for performance assessment.",
                    " in determining market position.",
                    " for accurate forecasting.",
                    " to provide actionable insights."
                ]
                sentence += random.choice(endings)
                sentences.append(sentence)
            
            text = " ".join(sentences)
        
        # Ensure text length constraints
        if len(text) < min_length:
            # Append additional content to reach minimum length
            while len(text) < min_length:
                additional_term = random.choice(terms)
                additional_text = f" Additionally, {additional_term} plays an important role in {category} analysis."
                text += additional_text
        
        if len(text) > max_length:
            # Truncate to maximum length at a sentence boundary
            sentences = text.split('. ')
            truncated_text = ''
            for sentence in sentences:
                if len(truncated_text) + len(sentence) + 2 <= max_length:
                    truncated_text += sentence + '. '
                else:
                    break
            text = truncated_text.strip()
        
        calibration_texts.append(text)
    
    # Ensure we have exactly the requested number of texts
    return calibration_texts[:count]


def load_or_generate_financial_calibration_dataset(
    file_path: Optional[str] = None,
    count: int = 100,
    force_regenerate: bool = False,
    categories: Optional[List[str]] = None,
    save: bool = True,
    seed: Optional[int] = None
) -> List[str]:
    """
    Load existing or generate new financial calibration dataset.
    
    Parameters
    ----------
    file_path : str, optional
        Path to existing calibration dataset (None to generate without saving)
    count : int, default=100
        Number of calibration texts to generate if creating a new dataset
    force_regenerate : bool, default=False
        Whether to force regeneration even if file exists
    categories : List[str], optional
        Categories to include (all if None)
    save : bool, default=True
        Whether to save the generated dataset
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    List[str]
        List of financial calibration texts
    """
    # Try to load existing file if provided and not forcing regeneration
    if file_path and os.path.exists(file_path) and not force_regenerate:
        try:
            logger.info(f"Loading existing financial calibration dataset from {file_path}")
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Check if it has the expected structure
                if isinstance(data, dict) and "calibration_texts" in data:
                    texts = data["calibration_texts"]
                    if isinstance(texts, list) and len(texts) > 0 and isinstance(texts[0], str):
                        logger.info(f"Loaded {len(texts)} calibration texts")
                        return texts
                
                # Fall back to simple list structure
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
                    logger.info(f"Loaded {len(data)} calibration texts")
                    return data
                
                logger.warning(f"Invalid calibration dataset format in {file_path}, regenerating")
        except Exception as e:
            logger.warning(f"Error loading calibration dataset: {str(e)}, regenerating")
    
    # Generate new calibration dataset
    logger.info(f"Generating new financial calibration dataset with {count} texts")
    texts = generate_calibration_texts(
        count=count,
        include_categories=categories,
        seed=seed
    )
    
    # Save if requested
    if file_path and save:
        try:
            logger.info(f"Saving financial calibration dataset to {file_path}")
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump({
                    "calibration_texts": texts,
                    "metadata": {
                        "count": len(texts),
                        "categories": categories or list(FINANCIAL_TERMS.keys()),
                        "generated_at": "2023-07-01T12:00:00Z",
                        "version": "1.0.0"
                    }
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving calibration dataset: {str(e)}")
    
    return texts


def create_enhanced_financial_calibration_dataset(
    count: int = 200,
    file_path: Optional[str] = None,
    include_standard_dataset: bool = True
) -> List[str]:
    """
    Create an enhanced financial calibration dataset for optimal INT8 quantization.
    
    This function combines generated financial texts with standard calibration data
    to create a comprehensive dataset for INT8 quantization of financial models.
    
    Parameters
    ----------
    count : int, default=200
        Number of financial-specific calibration texts to generate
    file_path : str, optional
        Path to save the calibration dataset
    include_standard_dataset : bool, default=True
        Whether to include standard calibration data for general language
    
    Returns
    -------
    List[str]
        Complete calibration dataset
    """
    # Generate financial-specific calibration texts
    logger.info(f"Creating enhanced financial calibration dataset with {count} texts")
    financial_texts = generate_calibration_texts(
        count=count,
        min_length=100,
        max_length=1000
    )
    
    # Combine with standard calibration data if requested
    if include_standard_dataset:
        # Import here to avoid circular imports
        try:
            from langchain_hana.gpu.calibration_datasets import load_calibration_dataset
            standard_texts = load_calibration_dataset(domain="general", count=100)
            logger.info(f"Adding {len(standard_texts)} standard calibration texts")
            all_texts = financial_texts + standard_texts
        except ImportError:
            logger.warning("Standard calibration dataset not available, using only financial texts")
            all_texts = financial_texts
    else:
        all_texts = financial_texts
    
    # Save if requested
    if file_path:
        try:
            logger.info(f"Saving enhanced financial calibration dataset to {file_path}")
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump({
                    "calibration_texts": all_texts,
                    "metadata": {
                        "count": len(all_texts),
                        "financial_count": len(financial_texts),
                        "standard_count": len(all_texts) - len(financial_texts),
                        "generated_at": "2023-07-01T12:00:00Z",
                        "version": "1.0.0"
                    }
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving enhanced calibration dataset: {str(e)}")
    
    return all_texts


def get_financial_calibration_dataset(
    count: int = 200,
    cache_dir: Optional[str] = None,
    force_regenerate: bool = False
) -> List[str]:
    """
    Get a financial calibration dataset for INT8 quantization.
    
    This is the main function to use when you need a calibration dataset
    for financial models.
    
    Parameters
    ----------
    count : int, default=200
        Number of calibration texts to include
    cache_dir : str, optional
        Directory to cache calibration datasets
    force_regenerate : bool, default=False
        Whether to force regeneration even if cached dataset exists
    
    Returns
    -------
    List[str]
        Financial calibration dataset
    """
    # Determine cache file path
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        file_path = os.path.join(cache_dir, f"financial_calibration_{count}.json")
    else:
        cache_dir = os.path.join(os.path.expanduser("~"), ".langchain_hana", "calibration")
        os.makedirs(cache_dir, exist_ok=True)
        file_path = os.path.join(cache_dir, f"financial_calibration_{count}.json")
    
    # Load or generate dataset
    return load_or_generate_financial_calibration_dataset(
        file_path=file_path,
        count=count,
        force_regenerate=force_regenerate
    )