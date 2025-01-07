from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

# Database connection setup
def get_db_connection():
    try:
        conn = psycopg2.connect(
            database="postgres",
            host="chopper.local",
            user="admin",
            password="admin123",
            port="5432"
        )
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail="Database connection failed")

# Ensure table exists
def initialize_database():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                id SERIAL PRIMARY KEY,
                car_count FLOAT NOT NULL,
                truck_count FLOAT NOT NULL,
                bike_count FLOAT NOT NULL,
                ped_count FLOAT NOT NULL,
                battery_percentage FLOAT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to initialize database")

initialize_database()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/count")
async def get_latest_from_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT * FROM readings
            ORDER BY id DESC
            LIMIT 1;
        """)
        latest_reading = cursor.fetchone()
        conn.close()
        
        if latest_reading:
            return {
                "Car_Count": latest_reading["car_count"],
                "Truck_Count": latest_reading["truck_count"],
                "Bike_Count": latest_reading["bike_count"],
                "Ped_Count": latest_reading["ped_count"],
                "Battery_Percentage": latest_reading["battery_percentage"]
            }
        else:
            return {
                "Car_Count": None,
                "Truck_Count": None,
                "Bike_Count": None,
                "Ped_Count": None,
                "Battery_Percentage": None
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch data from database")

@app.get("/history")
async def get_historical_data():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT * FROM readings
            ORDER BY id DESC
            LIMIT 100;
        """)
        readings = cursor.fetchall()
        conn.close()

        historical_data = [
            {
                "timestamp": reading["timestamp"].timestamp() * 1000,  # Convert to milliseconds
                "Car_Count": reading["car_count"],
                "Truck_Count": reading["truck_count"],
                "Bike_Count": reading["bike_count"],
                "Ped_Count": reading["ped_count"],
                "Battery_Percentage": reading["battery_percentage"]
            }
            for reading in readings
        ]
        return historical_data
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch historical data")

async def store_data(ccount: float, tcount: float, bcount: float, pcount: float, bpercent: float):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO readings (car_count, truck_count, bike_count, ped_count, battery_percentage)
            VALUES (%s, %s, %s, %s, %s);
        """, (ccount, tcount, bcount, pcount, bpercent))
        conn.commit()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to store data")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("vehcnt:app", host="0.0.0.0")
