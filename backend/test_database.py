from app.core.database import engine


try:
    with engine.connect() as connection:
        print("Database connection successful!")

except Exception as error:
    print("Database connection failed!")
    print(error)