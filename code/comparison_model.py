import sys, os, pandas as pd, matplotlib.pyplot as plt, numpy as np
from time import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main import load_and_standardize_data, cross_validate_model
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import joblib

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
colors = ['#6baed6', '#74c476', '#9e9ac8', '#fc9272']

print("Running all models with 5x20 cross validation")

X_red, y_red, f_red = load_and_standardize_data('red')
X_white, y_white, f_white = load_and_standardize_data('white')

models = [
    {'name': 'Linear Regression', 'red': LinearRegression(), 'white': LinearRegression()},
    {'name': 'XGBoost', 'red': XGBRegressor(n_estimators=100, random_state=42), 'white': XGBRegressor(n_estimators=100, random_state=42)},
    {'name': 'Random Forest', 'red': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), 'white': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)},
    {'name': 'SVM', 'red': SVR(kernel='rbf'), 'white': SVR(kernel='rbf')}
]

all_results = []
start = time()
for m in models:
    print(f"\n{m['name']}")
    for wine, X, y in [('Red', X_red, y_red), ('White', X_white, y_white)]:
        t0 = time()
        r = cross_validate_model(m[wine.lower()], X, y, wine, m['name'], cv_folds=5, n_repeats=20)
        all_results.append(r)
        m[wine.lower()].fit(X, y)
        joblib.dump(m[wine.lower()], f'models/{m["name"].lower().replace(" ", "_")}_{wine.lower()}.pkl')
        print(f"  {wine}: {time()-t0:.1f}s")

print(f"\nTotal: {time()-start:.1f}s")

results_df = pd.DataFrame(all_results)
results_df.to_csv('results/all_models_results.csv', index=False)
for m in models:
    name = m['name']
    results_df[results_df['model'] == name].to_csv(f'results/{name.lower().replace(" ", "_")}_cv_results.csv', index=False)

print("\nSaving feature importance")
for i, (name, col) in enumerate([('Linear Regression', 'coefficient'), ('XGBoost', 'importance'), ('Random Forest', 'importance')]):
    for wine, feat in [('red', f_red), ('white', f_white)]:
        if col == 'coefficient':
            imp_data = models[i][wine].coef_
        else:
            imp_data = models[i][wine].feature_importances_
        df = pd.DataFrame({'feature': feat, col: imp_data})
        df.to_csv(f'results/{name.lower().replace(" ", "_")}_{col}_{wine}.csv', index=False)

red = results_df[results_df['dataset']=='Red'].set_index('model')
white = results_df[results_df['dataset']=='White'].set_index('model')

print("\n" + "="*50)
print("RED WINE RESULTS")
print(red[['MAD', 'MSE', 'RMSE']].round(3))
print("\nWHITE WINE RESULTS")
print(white[['MAD', 'MSE', 'RMSE']].round(3))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Model Performance Comparison - All Metrics (5-fold CV, 20 repeats)', fontsize=14)

metrics = ['MAD', 'MSE', 'RMSE']
model_order = ['Random Forest', 'XGBoost', 'SVM', 'Linear Regression']
bar_width = 0.2
x = np.arange(len(metrics))

for col, (data, wine) in enumerate(zip([red, white], ['Red Wine', 'White Wine'])):
    ax = axes[col]
    for i, model in enumerate(model_order):
        if model in data.index:
            values = [data.loc[model][m] for m in metrics]
            bars = ax.bar(x + i*bar_width, values, bar_width, 
                         label=model, color=colors[i], edgecolor='black')
            
            # Add value labels
            for j, v in enumerate(values):
                ax.text(x[j] + i*bar_width, v*1.02, f'{v:.3f}', 
                       ha='center', va='bottom', fontsize=7, rotation=45)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    ax.set_title(wine)
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/model_comparison_all_metrics.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: figures/model_comparison_all_metrics.png")

imp = {}

for model in ['linear_regression', 'xgboost', 'random_forest']:
    for wine in ['red', 'white']:
        try:
            df = pd.read_csv(f'results/{model}_{"coefficient" if model=="linear_regression" else "importance"}_{wine}.csv')
            col = 'coefficient' if model == 'linear_regression' else 'importance'
            for _, row in df.iterrows():
                imp.setdefault(row['feature'], []).append(abs(row[col]))
        except:
            continue

top5 = pd.DataFrame([
    {'feature': f, 'importance': np.mean(v), 'std': np.std(v)}
    for f, v in imp.items()
]).sort_values('importance', ascending=False).head(5)

print("\nTOP 5 FEATURES (Avg across all models)")
print(top5[['feature', 'importance', 'std']].round(3).to_string(index=False))

plt.figure(figsize=(10, 4))
plt.barh(top5['feature'], top5['importance'], xerr=top5['std'], 
         color='#6baed6', edgecolor='black', capsize=3)
plt.xlabel('Average Importance')
plt.title('Top 5 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('figures/top5_features.png', dpi=150)
print("✓ Saved: figures/top5_features.png")

print("\n" + "="*50)
print("BEST MODEL BY METRIC")
print("="*50)
for wine, data in [('Red', red), ('White', white)]:
    print(f"\n{wine} WINE:")
    for metric in ['MAD', 'MSE', 'RMSE']:
        best = data[metric].idxmin()
        print(f"  {metric}: {best} ({data.loc[best, metric]:.3f})")

print("\nDone!")