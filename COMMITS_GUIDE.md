## Commit Guide for ML Projects

This guide enforces a consistent history, crucial for debugging, collaboration, and MLOps. All commits must follow the Conventional Commits structure: type(scope): subject.

### 1. Commit Message Structure

| Part | Rule | Example | Purpose |
| :--- | :--- | :--- | :--- |
| Type | Mandatory. Defines the *intent* of the change (e.g., adding a feature, fixing a bug). | feat, fix, refactor, test | Categorizes the commit for easy searching. |
| Scope | Optional. Provides *context* (e.g., the specific file or module affected). | (pipeline), (api), (src/features) | Narrows down the location of the change. |
| Subject | Mandatory. A concise, imperative sentence summarizing the change. | Implement robust ColumnTransformer structure | Clearly describes the change's action. |

### 2. Standard Commit Types (The "What")

| Type | When to Use | Example Commit Message | ML Workflow Stage |
| :--- | :--- | :--- | :--- |
| feat | A new **feature** or significant model enhancement. | feat(pipeline): Add XGBoost model to baseline comparison | Modeling, Deployment |
| fix | A bug fix or correction to incorrect logic/data. | fix(data): Remove duplicate rows and handle division by zero | EDA, Preprocessing, API |
| refactor | Restructuring code without changing functionality (e.g., moving code to src/). | refactor(src): Move custom feature logic into features.py | Code Organization |
| docs | Changes to documentation, READMEs, or docstrings. | docs: Update README with final AUC and deployment guide | Finalization, Cleanup |
| test | Adding missing tests or correcting existing tests. | test(unit): Add unit test for LTI ratio edge cases | Quality Assurance |
| build | Changes that affect the build system or external dependencies. | build(docker): Update Dockerfile to Python 3.10 slim base | Containerization |
| chore | Routine tasks that don't affect code logic (e.g., updating .gitignore). | chore: Clean up unnecessary temporary notebooks | Maintenance |
| style | Formatting changes (whitespace, indentation, semicolons). | style: Apply Black formatting to all Python files | Code Cleanup |

### 3. When to Commit (The "When")

Commit often, ensuring each commit captures one complete, logical change.

| Workflow Stage | Action Completed | Recommended Commit Message |
| :--- | :--- | :--- |
| Setup & Integrity | Initial project structure and removal of ID columns/duplicates. | fix(data): Remove duplicate rows and drop Loan_ID column |
| Feature Engineering | Finalizing the logic for a custom feature (e.g., LTI Ratio function). | feat(features): Introduce create_loan_to_income_ratio function |
| Preprocessing | Defining the complex ColumnTransformer object. | feat(pipeline): Define all routing for scaling, binning, and encoding |
| Modeling | Adding a new model (e.g., Random Forest) to the comparison. | feat(baselines): Add Random Forest classifier to evaluation |
| Deployment Prep | Setting up the Docker image or API endpoint logic. | build(docker): Add initial Dockerfile for FastAPI service |
| Finalization | Saving the final, best model artifact. | feat(deploy): Save final production_pipeline.pkl artifact |