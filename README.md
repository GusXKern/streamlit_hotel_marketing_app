# 🏨 Causal Campaign Targeting & ROI Optimizer

A production-grade data science project that shifts hotel marketing from traditional predictive analytics (who is likely to book?) to **causal inference** (who can be *persuaded* to book by an email discount?). 

This project bridges the gap between raw data exploration, custom causal experiment design, machine learning modeling, and an interactive **Streamlit web application** designed for marketing and finance teams.

---

## 🧭 Project Workflow & Technical Analysis

### Phase 1: Exploratory Data Analysis & Baseline (Guided Foundation)
* **Data Foundation:** Started by exploring hotel booking trends, seasonal lead times, market segments, and cancellation patterns following an instructional video framework.
* **Feature Engineering:** Cleaned raw attributes, handled missing values, and isolated key drivers of baseline customer behavior (e.g., country of origin, deposit types, distribution channels, and room allocation discrepancies).

### Phase 2: Causal Experiment Design & Data Generation (Independent Extension)
* **The Causal Problem:** Standard predictive models often target "sure things" (customers who would book anyway) or "lost causes" (customers who won't book regardless of incentives). Giving discounts to sure-things wastes precious marketing margin.
* **Simulating Intervention:** Designed and generated a custom experimental dataset introducing a randomized treatment (sending a targeted discount email vs. control/no email). 
* **Target Variable Construction:** Tracked the true counterfactual outcomes to measure individual treatment effects—separating baseline intent from actual persuasive impact.

### Phase 3: LightGBM T-Learner Architecture
* **Two-Model Strategy:** Implemented a **T-Learner** (Two-Model Learner) framework utilizing **LightGBM** gradient-boosted decision trees under the hood.
  1. *Treatment Model ($M_1$):* Trained exclusively on customers who received the discount offer to learn behavior under intervention.
  2. *Control Model ($M_0$):* Trained exclusively on the control group to learn baseline behavior without intervention.
* **Countefactual Scoring:** For any new customer, the model simultaneously simulates both worlds: $P(\text{Booking} \mid X, \text{Treated})$ and $P(\text{Booking} \ X, \ \text{Control})$.
* **CATE Calculation:** Subtracted the two probabilities to compute the individual **Conditional Average Treatment Effect (CATE / Uplift Score)**:
  $$\text{Uplift} = P(\text{Treatment}) - P(\text{Control})$$

### Phase 4: Productionization & Decision-Making (Streamlit App)
To transition this from a Jupyter notebook to a business tool, I built a production application (`hotel_app.py`) featuring:
* **Automated Data Preprocessing Pipeline:** Ingests raw incoming CSV booking sheets from marketing teams, automatically handling one-hot encoding (e.g., `country_PRT`, `market_segment_Groups`, `deposit_type_Non Refund`) and aligning the schema strictly to the model's expected feature array (`x_names.pkl`).
* **Dynamic Campaign Economics:** Allows finance and marketing leads to interactively adjust campaign parameters—such as **Average Booking Revenue** and **Offer Discount Cost**—in real time.
* **Net Financial ROI Ranking:** Translates abstract uplift probabilities into hard dollars using the economic formula:
  $$\text{Expected Net ROI} = (\text{Uplift Score} \times \text{Avg Revenue}) - (\text{Predicted Treatment Prob} \times \text{Discount Cost})$$
* **Actionable Export Engine:** Filters out unprofitable accounts and generates a prioritized, downloadable CSV containing customer contact details (`name`, `email`, `phone-number`) strictly targeting high-ROI "persuadables."

---

## 🚀 Tech Stack
* **Language & Core Libraries:** Python, Pandas, NumPy, Scikit-Learn, Joblib
* **Causal Machine Learning:** LightGBM
* **Production UI:** Streamlit
