"""
Basic usage example for CCGL Analytics
"""

from ccgl_analytics import CCGLAnalyzer
import pandas as pd
import numpy as np

def main():
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'x': np.random.normal(0, 1, 100),
        'y': np.random.normal(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Initialize analyzer
    analyzer = CCGLAnalyzer()
    
    # Perform analysis
    results = analyzer.analyze_data(data=data, analysis_type=['clustering', 'quality'])
    
    print("Analysis completed!")
    print(f"Results: {results}")

if __name__ == "__main__":
    main()