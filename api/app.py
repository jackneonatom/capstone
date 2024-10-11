from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import motor.motor_asyncio
from datetime import datetime

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://localhost:27017")
db = client.vechilecount

@app.get("/count")
async def get_latest_from_db():
    try:
        latest_reading = await db["readings"].find_one(sort=[("_id", -1)])
        if latest_reading:
            return {"Car_Count": latest_reading["Car_Count"], "Truck_Count": latest_reading["Truck_Count"], "Bike_Count": latest_reading["Bike_Count"], "Ped_Count": latest_reading["Ped_Count"], "Battery_Percentage": latest_reading["Battery_Percentage"]}
        else:
            return {"Car_Count": None, "Truck_Count": None, "Bike_Count": None, "Ped_Count": None , "Battery_Percentage":None}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch data from database")

@app.get("/history")
async def get_historical_data():
    try:
        readings = await db["readings"].find().sort("_id", -1).to_list(length=100)  # Adjust length as needed
        historical_data = [
            {
                "timestamp": reading["_id"].generation_time.timestamp() * 1000,  # Convert to milliseconds
                "Car_Count": reading["Car_Count"],
                "Truck_Count": reading["Truck_Count"], 
                "Bike_Count": reading["Bike_Count"], 
                "Ped_Count": reading["Ped_Count"], 
                "Battery_Percentage": reading["Battery_Percentage"]
                           
            }
            for reading in readings
        ]
        return historical_data
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch historical data")


async def store_data(ccount: float, tcount: float, bcount: float, pcount: float, bpercent: float):
    count_state = {
        "Car_Count": ccount,
        "Truck_Count": tcount,
        "Bike_Count": bcount,
        "Ped_Count": pcount,
        "Battery_Percentage": bpercent,
        
    }

    result = await db["readings"].insert_one(count_state)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0")