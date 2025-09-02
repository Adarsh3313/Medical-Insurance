{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "5e81090f-8fc9-4e3f-bf45-b363ee694efd",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-09-02 14:18:30.106 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\adars\\anaconda\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-09-02 14:18:30.113 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import pickle\n",
    "\n",
    "# -----------------------------\n",
    "# Load trained model\n",
    "# -----------------------------\n",
    "with open(\"model.pkl\", \"rb\") as f:\n",
    "    model = pickle.load(f)\n",
    "\n",
    "st.title(\" Medical Insurance Cost Prediction\")\n",
    "\n",
    "st.write(\"Fill in the details below to estimate your medical expenses:\")\n",
    "\n",
    "# -----------------------------\n",
    "# User Inputs\n",
    "# -----------------------------\n",
    "age = st.number_input(\"Age\", min_value=18, max_value=100, step=1)\n",
    "sex = st.selectbox(\"Sex\", [\"male\", \"female\"])\n",
    "bmi = st.number_input(\"BMI (Body Mass Index)\", min_value=10.0, max_value=60.0, step=0.1)\n",
    "children = st.number_input(\"Number of Children\", min_value=0, max_value=10, step=1)\n",
    "smoker = st.selectbox(\"Smoker\", [\"yes\", \"no\"])\n",
    "region = st.selectbox(\"Region\", [\"northeast\", \"northwest\", \"southeast\", \"southwest\"])\n",
    "\n",
    "# -----------------------------\n",
    "# Convert input into dataframe (same encoding as training)\n",
    "# -----------------------------\n",
    "input_data = pd.DataFrame(\n",
    "    {\n",
    "        \"age\": [age],\n",
    "        \"bmi\": [bmi],\n",
    "        \"children\": [children],\n",
    "        \"sex_male\": [1 if sex == \"male\" else 0],\n",
    "        \"smoker_yes\": [1 if smoker == \"yes\" else 0],\n",
    "        \"region_northwest\": [1 if region == \"northwest\" else 0],\n",
    "        \"region_southeast\": [1 if region == \"southeast\" else 0],\n",
    "        \"region_southwest\": [1 if region == \"southwest\" else 0],\n",
    "    }\n",
    ")\n",
    "\n",
    "# -----------------------------\n",
    "# Prediction\n",
    "# -----------------------------\n",
    "if st.button(\"Predict Expense\"):\n",
    "    prediction = model.predict(input_data)[0]\n",
    "    st.success(f\" Estimated Medical Expense: ${prediction:,.2f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fb7e514a-77ce-40c6-b307-4c5f1b3e3fe2",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
