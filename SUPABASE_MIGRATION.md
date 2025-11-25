# Database Migration & Crash Fix Walkthrough

We successfully migrated the backend to Supabase and resolved a critical server crash.

## 1. Database Migration
We moved from a local SQLite database to a cloud-hosted PostgreSQL database on Supabase.

### Changes
- **Configuration**: Created `.env` with Supabase credentials (`DB_USER`, `DB_PASSWORD`, etc.).
- **Dependencies**: Added `psycopg2-binary` and `python-dotenv`.
- **Code**: Refactored `app/database.py` to use `DATABASE_URL` from environment variables.

## 2. The Crash: `free(): invalid pointer`
After migration, the server crashed with `free(): invalid pointer` whenever a prediction was requested or sometimes at startup.

### The Cause
This is a known conflict between **TensorFlow** and **psycopg2** (the PostgreSQL driver). Both libraries use C extensions that link against system libraries (like SSL or OpenMP). When imported in a specific order, they can conflict, leading to memory corruption and crashes.

### The Solution
We fixed this by **reordering the imports** to ensure the database driver is initialized *before* TensorFlow.

#### `app/main.py`
```python
# Import database FIRST to avoid conflict with TensorFlow
from .database import engine, Base
from .models import sql_models

from .services import prediction  # Imports TensorFlow
```

#### `app/services/prediction.py`
```python
# Import database FIRST to avoid conflict with TensorFlow
from ..database import SessionLocal
from ..models.sql_models import ImageLog

import tensorflow as tf  # Now safe to import
```

## 3. Improved Logging
We also cleaned up the logs:
- **Silenced TensorFlow**: Moved `os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'` to the very top of `app/main.py`.
- **Connection Check**: Implemented a FastAPI `lifespan` handler to print a green success message on startup.

```text
✔ Supabase DB Connected Successfully
```

## Verification
The server now starts correctly, connects to Supabase, and processes predictions without crashing.

```bash
INFO:     Started server process [34542]
INFO:     Waiting for application startup.
✔ Supabase DB Connected Successfully
INFO:     Application startup complete.
Received request for prediction
Dog detected!
Prediction completed
INFO:     127.0.0.1:50958 - "POST /predict HTTP/1.1" 200 OK
```
