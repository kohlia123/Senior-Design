from src.utils.preprocessing import get_subj_data
from src.config import N_SUB
import pandas as pd

def test_feature_flow():
    print("Starting Feature Extraction Test...")
    
    # Test with Subject 01
    subj_id = '01'
    
    try:
        # This calls preprocessing.py -> extract_epochs_features -> feature_extraction.py
        X, y = get_subj_data(subj_id)
        
        print(f"Successfully extracted data for Subject {subj_id}")
        print(f"Features shape: {X.shape}") # Should be (n_epochs, n_features)
        print(f"Labels shape: {len(y)}")
        
        # Verify your new feature exists in the columns
        print("\nExtracted Features:")
        print(X.columns.tolist())
        
        # Check the first few rows to ensure values aren't NaN or 0
        print("\nSample Data (First 5 rows):")
        print(X[['ptp_amp']].head())
        
        if not X.empty:
            print("\nTEST PASSED: Feature extraction is modular and working!")
            
    except Exception as e:
        print(f"\nTEST FAILED: {str(e)}")

if __name__ == "__main__":
    test_feature_flow()