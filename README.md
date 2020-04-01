# Flask-API-for-store-face-embeddings
- After running the app go to localhost:5000
- Then follow the swagger UI.

# Database Introduction

It has two collections: USER, EMBD:.
USER:
- id:(string)
- userId:(string)
- password:(string)

EMBD:
- id:(string)  (common id for same user in both collection)
- embeddings:(array of size 128)
