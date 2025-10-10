# 🚚 Delivery_ML_WebApp

A **mini machine learning project** built using the **Flask framework**, designed to integrate trained models and present their functionalities through a clean **web interface**.

---

## 📖 Project Background

This project simulates a collaboration with a major **intra-city logistics company in India**, a leading player in the $40 billion logistics sector.  
Our client connects restaurants and customers through an optimized delivery network, employing over **150,000 driver-partners** and serving more than **5 million customers**.

The company’s main challenge is to provide customers with **accurate delivery time estimates**, leveraging data on orders, locations, and partner availability.  
This project aims to model that process using **machine learning regression** techniques deployed via Flask.

---

## 🎯 Associated Task

### 🔹 Regression
- **Prediction of delivery time estimation** based on operational and order-related features.

---

## 🧾 Data Description

Each row in the dataset represents a **unique delivery**, and each column a **feature** describing that delivery.

| Feature | Description |
|----------|--------------|
| `market_id` | Integer ID for the market where the restaurant is located |
| `created_at` | Timestamp when the order was placed |
| `actual_delivery_time` | Timestamp when the order was delivered |
| `store_primary_category` | Category of the restaurant |
| `order_protocol` | Integer code describing how the order was placed (e.g., through app, call, pre-booked) |
| `total_items_subtotal` | Final price of the order |
| `num_distinct_items` | Number of distinct items in the order |
| `min_item_price` | Price of the cheapest item in the order |
| `max_item_price` | Price of the most expensive item in the order |
| `total_onshift_partners` | Number of delivery partners available when the order was placed |
| `total_busy_partners` | Number of delivery partners already on other tasks |
| `total_outstanding_orders` | Number of orders pending fulfillment at that moment |

---

## 💡 Project Objectives

### 🧭 Business Objectives
1. **Improve customer satisfaction** – Provide accurate delivery time estimates to reduce uncertainty and enhance user experience.  
2. **Optimize driver allocation** – Efficiently assign available delivery partners to minimize delays and maximize completed deliveries.  
3. **Enhance marketing efficiency** – Tailor promotions based on customer segments and order behaviors.

### 🧮 Data Science Objectives
1. **Predict estimated delivery time** (regression problem) using features related to order details, location, and partner availability.  
2. **Detect delivery delays** through feature engineering and classification.  
3. **Segment customers** into three behavioral groups via feature engineering:  
   - 🧍‍♂️ **Frequent small-order clients** → Encourage through loyalty programs.  
   - 🏢 **Corporate bulk clients** → Offer long-term contracts.  
   - 💎 **Occasional high-value clients** → Target with premium offers.

---

## ⚙️ Tech Stack

- **Python**, **Flask**, **pandas**, **scikit-learn**
- **HTML/CSS** for front-end visualization
- **Pickle** / **joblib** for model serialization
- **Jupyter Notebook** for preprocessing and analysis

---

## 🚀 How It Works

1. Data preprocessing and feature engineering performed in Jupyter Notebook.  
2. Regression model trained to predict delivery time using historical data.  
3. Flask app imports the trained model and serves predictions through a simple web UI.

---

⭐ *“Delivering insight, one prediction at a time.”*
