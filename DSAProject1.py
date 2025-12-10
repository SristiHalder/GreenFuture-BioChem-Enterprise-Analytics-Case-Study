
# ---------------------------------------------
# 1. Import Libraries
# ---------------------------------------------
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------------------------
# 2. Set File Paths
# ---------------------------------------------
path_rd   = "/Users/sristihalder/Downloads/RD_Pipeline.xlsx"
path_sales = "/Users/sristihalder/Downloads/Sales_Pipeline (2).xlsx"
path_prod = "/Users/sristihalder/Downloads/Product_Master (1).xlsx"
path_mfg  = "/Users/sristihalder/Downloads/Manufacturing_Production (1).xlsx"
path_proc = "/Users/sristihalder/Downloads/SupplyChain_Procurement (2).xlsx"

out_root = "/Users/sristihalder/Desktop/GreenFuture_Project_All"
os.makedirs(out_root, exist_ok=True)

# ---------------------------------------------
# 3. Visualization Configuration (LaTeX Style)
# ---------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})
sns.set_theme(style="whitegrid")
palette_main = ["#1B263B", "#415A77", "#778DA9", "#E0E1DD"]

# ---------------------------------------------
# 4. Load All Datasets
# ---------------------------------------------
rd = pd.read_excel(path_rd)
sales = pd.read_excel(path_sales)
product = pd.read_excel(path_prod)
mfg = pd.read_excel(path_mfg)
proc = pd.read_excel(path_proc)

datasets = {
    "R&D": rd,
    "Sales": sales,
    "Product": product,
    "Manufacturing": mfg,
    "Procurement": proc
}

# ---------------------------------------------
# 5. Data Cleaning & Standardization
# ---------------------------------------------
for name, df in datasets.items():
    df.columns = df.columns.str.strip()
    df.drop_duplicates(inplace=True)
    print(f"{name} dataset shape: {df.shape}")

# Fill missing where appropriate
mfg.fillna(0, inplace=True)
proc.fillna(0, inplace=True)

# ---------------------------------------------
# 6. Create Output Folders for Each Section
# ---------------------------------------------
sections = ["SectionIII", "SectionIV", "SectionV", "SectionVI", "Appendix"]
for sec in sections:
    os.makedirs(f"{out_root}/{sec}", exist_ok=True)

# ============================================================
# SECTION III – DESCRIPTIVE ANALYTICS
# ============================================================

# Summary stats
print("=== Summary Statistics ===")
for name, df in datasets.items():
    print(f"\n{name} Dataset:")
    print(df.describe(include='all').transpose().head(10))

# R&D Funnel Visualization
plt.figure(figsize=(5,4))
sns.countplot(y="Stage", data=rd, order=rd["Stage"].value_counts().index, palette=palette_main)
plt.title("R&D Project Stage Distribution")
plt.xlabel("Count")
plt.ylabel("Stage")
plt.tight_layout()
plt.savefig(f"{out_root}/SectionIII/III1_RD_Funnel.png", dpi=300)
plt.close()

# Sales Funnel Visualization
plt.figure(figsize=(5,4))
sns.countplot(y="Stage", data=sales, order=sales["Stage"].value_counts().index, palette=palette_main)
plt.title("Sales Opportunity Stage Distribution")
plt.xlabel("Count")
plt.ylabel("Stage")
plt.tight_layout()
plt.savefig(f"{out_root}/SectionIII/III2_Sales_Funnel.png", dpi=300)
plt.close()

# Sales Product Interest
plt.figure(figsize=(6,4))
top_products = sales["Product_Interest"].value_counts().head(10)
sns.barplot(y=top_products.index, x=top_products.values, palette=palette_main)
plt.title("Top Product Interests (Sales Pipeline)")
plt.xlabel("Count")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig(f"{out_root}/SectionIII/III3_ProductInterest.png", dpi=300)
plt.close()

# Manufacturing Yield by Plant
plt.figure(figsize=(6,4))
sns.barplot(x="Plant_Code", y="Yield (%)", data=mfg, palette=palette_main)
plt.title("Average Yield (%) by Plant")
plt.tight_layout()
plt.savefig(f"{out_root}/SectionIII/III4_Yield_byPlant.png", dpi=300)
plt.close()

# Procurement On-time vs Late
plt.figure(figsize=(5,4))
sns.countplot(x="On_Time (Y/N)", data=proc, palette=["#E69F00", "#56B4E9"])
plt.title("Procurement: On-Time vs Late Deliveries")
plt.xlabel("")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{out_root}/SectionIII/III5_OnTime_vs_Late.png", dpi=300)
plt.close()

# ============================================================
# SECTION IV – DIAGNOSTIC ANALYTICS
# ============================================================

mfg["Cost_Variance_$"] = mfg["Actual_Cost_per_MT ($)"] - mfg["Std_Cost_per_MT ($)"]
mfg["Cost_Variance_%"] = (mfg["Cost_Variance_$"] / mfg["Std_Cost_per_MT ($)"]) * 100
mfg["Efficiency_Ratio"] = mfg["Actual_Quantity (MT)"] / mfg["Planned_Quantity (MT)"]

summary = mfg.groupby("Plant_Code").agg(
    Yield_mean=("Yield (%)", "mean"),
    CostVar_mean=("Cost_Variance_%", "mean"),
    EffRatio_mean=("Efficiency_Ratio", "mean")
).reset_index()

print("\n===== Diagnostic Summary =====")
print(summary)

# Boxplot of Yield
plt.figure(figsize=(6,4))
sns.boxplot(x="Plant_Code", y="Yield (%)", data=mfg, palette=palette_main)
plt.title("Yield Distribution by Plant")
plt.savefig(f"{out_root}/SectionIV/IV1_Yield_byPlant.png", dpi=300)
plt.close()

# Yield vs Cost Variance
plt.figure(figsize=(5,4))
sns.regplot(x="Yield (%)", y="Cost_Variance_%", data=mfg, color=palette_main[1])
plt.title("Yield vs Cost Variance (%)")
plt.savefig(f"{out_root}/SectionIV/IV2_Yield_vs_CostVariance.png", dpi=300)
plt.close()

# Avg Cost Variance by Plant
plt.figure(figsize=(5,4))
sns.barplot(x="Plant_Code", y="CostVar_mean", data=summary, palette=palette_main)
plt.title("Average Cost Variance (%) by Plant")
plt.savefig(f"{out_root}/SectionIV/IV3_CostVariance_byPlant.png", dpi=300)
plt.close()

# Correlation Matrix
corr_vars = ["Yield (%)","Std_Cost_per_MT ($)","Actual_Cost_per_MT ($)","Cost_Variance_%","Efficiency_Ratio"]
corr_matrix = mfg[corr_vars].corr(method="pearson")
sns.heatmap(corr_matrix, annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation Matrix — Production Efficiency Drivers")
plt.savefig(f"{out_root}/SectionIV/IV4_CorrelationMatrix.png", dpi=300)
plt.close()

# ============================================================
# SECTION V – SUSTAINABILITY & GROWTH
# ============================================================

proc["Emission_Intensity"] = proc["CO2_Emissions (kg/MT)"] / proc["Qty (MT)"]
supplier_summary = proc.groupby("Supplier_Name").agg(
    Total_Emissions=("CO2_Emissions (kg/MT)","sum"),
    Mean_Intensity=("Emission_Intensity","mean"),
    OnTime_Rate=("On_Time (Y/N)", lambda x: (x=="Y").mean())
).reset_index()
supplier_summary["OnTime_%"] = supplier_summary["OnTime_Rate"] * 100

# CO2 by Supplier
plt.figure(figsize=(6,4))
sns.barplot(y="Supplier_Name", x="Total_Emissions", data=supplier_summary, palette=palette_main)
plt.title("Total CO2 Impact by Supplier")
plt.xlabel("Total Emissions (kg)")
plt.tight_layout()
plt.savefig(f"{out_root}/SectionV/V1_CO2_bySupplier.png", dpi=300)
plt.close()

# On-Time vs Late
plt.figure(figsize=(6,4))
sns.histplot(proc["On_Time (Y/N)"], palette=palette_main)
plt.title("On-Time vs Late Deliveries")
plt.tight_layout()
plt.savefig(f"{out_root}/SectionV/V2_OnTime_vs_Late.png", dpi=300)
plt.close()

# R&D Revenue by Industry
plt.figure(figsize=(6,4))
sns.barplot(y="Target_Industry", x="Est_Annual_Revenue ($M)", data=rd, palette=palette_main)
plt.title("Estimated Annual Revenue by Industry")
plt.tight_layout()
plt.savefig(f"{out_root}/SectionV/V3_RD_Revenue_byIndustry.png", dpi=300)
plt.close()

# Linear Forecast
rd["Year"] = pd.to_datetime(rd["Est_Launch_Date"]).dt.year
yearly = rd.groupby("Year")["Est_Annual_Revenue ($M)"].sum().reset_index()
X, y = yearly["Year"].values.reshape(-1,1), yearly["Est_Annual_Revenue ($M)"].values
model = LinearRegression().fit(X,y)
future = pd.DataFrame({"Year": range(2024,2031)})
future["Forecast"] = model.predict(future[["Year"]])
plt.figure(figsize=(6,4))
plt.plot(yearly["Year"], y, label="Actual", marker="o")
plt.plot(future["Year"], future["Forecast"], label="Forecast", linestyle="--")
plt.title("Projected R&D Revenue Growth (2024–2030)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{out_root}/SectionV/V4_RD_Revenue_Forecast.png", dpi=300)
plt.close()


proc_summary = proc.groupby("Supplier_Name").agg(
    OnTime_Rate=("On_Time (Y/N)", lambda x: (x=="Y").mean()),
    Mean_Intensity=("CO2_Emissions (kg/MT)", "mean")
).reset_index()

plt.figure(figsize=(6,4))
sns.regplot(x="OnTime_Rate", y="Mean_Intensity", data=proc_summary, color=palette_main[1])
plt.title("Supplier Reliability vs Emission Intensity")
plt.tight_layout()
plt.savefig(f"{out_root}/SectionVI/VI1_Supplier_Reliability_Emissions.png", dpi=300)
plt.close()

themes = ["Operational Efficiency","Sustainable Procurement","Logistics Optimization","Innovation Growth"]
impact = [8,7,6,9]
feasibility = [7,6,8,8]

plt.figure(figsize=(5,4))
sns.scatterplot(x=feasibility, y=impact, color=palette_main[1])
for i,t in enumerate(themes):
    plt.text(feasibility[i]+0.1, impact[i], t, fontsize=9)
plt.xlabel("Feasibility")
plt.ylabel("Impact")
plt.title("Strategic Prioritization Matrix")
plt.tight_layout()
plt.savefig(f"{out_root}/SectionVI/VI_Impact_vs_Feasibility.png", dpi=300)
plt.close()

sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Appendix: Correlation Matrix (Replotted)")
plt.tight_layout()
plt.savefig(f"{out_root}/Appendix/Appendix_CorrelationMatrix.png", dpi=300)
plt.close()

sns.regplot(x="Yield (%)", y="Cost_Variance_%", data=mfg, color=palette_main[2])
plt.title("Appendix: Yield vs Cost Variance (Regression)")
plt.tight_layout()
plt.savefig(f"{out_root}/Appendix/Appendix_Yield_vs_CostVariance.png", dpi=300)
plt.close()

sns.regplot(x="OnTime_Rate", y="Mean_Intensity", data=proc_summary, color=palette_main[2])
plt.title("Appendix: Supplier Reliability vs Emission Intensity")
plt.tight_layout()
plt.savefig(f"{out_root}/Appendix/Appendix_Supplier_Reliability_Emissions.png", dpi=300)
plt.close()

print("All sections executed successfully. Visualizations and tables saved to output folder.")
