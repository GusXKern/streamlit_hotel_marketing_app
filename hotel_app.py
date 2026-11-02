import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Hotel Campaign & Causal ROI Optimizer",
    page_icon="🏨",
    layout="wide"
)

# --- LOAD ARTIFACTS WITH CACHING ---
@st.cache_resource
def load_artifacts():
    model_path = 'artifacts/t_learner_hotel_model.pkl'
    features_path = 'artifacts/x_names.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        return None, None
    
    model = joblib.load(model_path)
    x_names = joblib.load(features_path)
    return model, x_names

tm_model, x_names = load_artifacts()

# --- TITLE & DESCRIPTION ---
st.title("🏨 Causal Campaign Targeting & ROI Optimizer")
st.markdown(
    "Upload incoming raw customer booking data to score individual treatment effects (CATE) "
    "and generate a ranked, high-ROI target list for your discount email campaign."
)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("1. Campaign Economics")
avg_revenue = st.sidebar.number_input("Average Booking Revenue ($)", value=1500, step=50, help="Baseline revenue per successful booking.")
discount_cost = st.sidebar.number_input("Offer Discount Cost ($)", value=150, step=10, help="Cost of the discount when accepted.")

st.sidebar.header("2. Upload Incoming Data")
uploaded_file = st.sidebar.file_uploader("Upload Raw Bookings CSV", type=["csv"])

# --- MAIN LOGIC ---
if tm_model is None or x_names is None:
    st.error(
        "⚠️ Model artifacts not found! Please make sure `t_learner_hotel_model.pkl` and `x_names.pkl` "
        "are saved inside the `artifacts/` folder."
    )
elif uploaded_file is not None:
    # Load raw user data
    df_raw = pd.read_csv(uploaded_file)
    
    with st.spinner("Preprocessing raw features and scoring customer uplift..."):
        df_processed = df_raw.copy()
        
        # 1. Recreate engineered features & one-hot encodings to match training schema
        # Room difference feature
        if 'reserved_room_type' in df_processed.columns and 'assigned_room_type' in df_processed.columns:
            df_processed['different_room_assigned'] = (df_processed['reserved_room_type'] != df_processed['assigned_room_type']).astype(int)
        else:
            df_processed['different_room_assigned'] = 0
            
        # Lead time transformation
        if 'lead_time' in df_processed.columns:
            df_processed['new_lead_time'] = df_processed['lead_time']
        else:
            df_processed['new_lead_time'] = 0

        # Country binary flags
        if 'country' in df_processed.columns:
            df_processed['country_PRT'] = (df_processed['country'] == 'PRT').astype(int)
            df_processed['country_other'] = (df_processed['country'] != 'PRT').astype(int)
        else:
            df_processed['country_PRT'] = 0
            df_processed['country_other'] = 0

        # Standard categorical dummy variables
        cat_columns = ['hotel', 'deposit_type', 'market_segment', 'distribution_channel', 'customer_type']
        df_encoded = pd.get_dummies(df_processed, columns=[c for c in cat_columns if c in df_processed.columns], drop_first=False)

        # 2. Strictly align columns with model's expected x_names (adds missing as 0, drops extras)
        for col in x_names:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
                
        X_inference = df_encoded[x_names].values

        # 3. Run inference using the trained T-Learner model
        tm_pred = tm_model.predict(X_inference)
        
        df = df_raw.copy() # Keep raw columns (names, emails) for final marketing output
        
        # Bulletproof handling for model prediction shapes
        if isinstance(tm_pred, np.ndarray) and tm_pred.ndim == 1:
            df['uplift_score'] = tm_pred
            df['predicted_treatment_prob'] = np.clip(tm_pred, 0, 1)
        elif hasattr(tm_model, '_classes') and len(tm_model._classes) >= 2:
            pred_df = pd.DataFrame(tm_pred, columns=tm_model._classes)
            control_col = 'control' if 'control' in pred_df.columns else pred_df.columns[0]
            treatment_col = [c for c in pred_df.columns if c != control_col][0]
            df['predicted_treatment_prob'] = pred_df[treatment_col]
            df['predicted_control_prob'] = pred_df[control_col]
            df['uplift_score'] = df['predicted_treatment_prob'] - df['predicted_control_prob']
        else:
            df['uplift_score'] = tm_pred[:, 0]
            df['predicted_treatment_prob'] = np.clip(tm_pred[:, 0], 0, 1)
        
        # 4. Expected Net Financial ROI per customer
        df['expected_net_roi'] = (df['uplift_score'] * avg_revenue) - (df['predicted_treatment_prob'] * discount_cost)
        
        # Rank by Expected Net ROI descending
        df_ranked = df.sort_values(by='expected_net_roi', ascending=False).reset_index(drop=True)
        
        # Filter accounts with positive projected ROI
        target_accounts = df_ranked[df_ranked['expected_net_roi'] > 0]
        
    # --- METRICS BANNER ---
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Bookings Scored", f"{len(df):,}")
    col2.metric("Recommended Email Targets", f"{len(target_accounts):,}", delta=f"{len(target_accounts)/len(df):.1%} of total")
    col3.metric("Projected Campaign Net ROI", f"${target_accounts['expected_net_roi'].sum():,.2f}")
    col4.metric("Avg. Expected Return / Target", f"${target_accounts['expected_net_roi'].mean():,.2f}" if len(target_accounts) > 0 else "$0.00")
    
    # --- INTERACTIVE VIEW TOGGLE ---
    st.markdown("### 📋 Ranked Target Account List")
    filter_option = st.radio(
        "Display View:", 
        ["Show Positive ROI Targets Only (Recommended)", "Show All Scored Accounts"],
        horizontal=True
    )
    
    display_df = target_accounts if "Positive" in filter_option else df_ranked
    
    # Display DataFrame with essential columns prioritized if available
    priority_cols = [c for c in ['name', 'email', 'phone-number', 'lead_time', 'market_segment', 'uplift_score', 'expected_net_roi'] if c in display_df.columns]
    other_cols = [c for c in display_df.columns if c not in priority_cols]
    
    st.dataframe(
        display_df[priority_cols + other_cols],
        use_container_width=True
    )
    
    # --- DOWNLOAD BUTTON ---
    csv_data = target_accounts.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Actionable Email List (.csv)",
        data=csv_data,
        file_name="targeted_hotel_email_campaign.csv",
        mime="text/csv",
        help="Downloads only the accounts where expected net campaign ROI is greater than zero, including customer contact info."
    )
else:
    st.info("👈 Please upload your raw customer bookings CSV file in the sidebar to run the campaign targeting engine.")