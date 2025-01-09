import httpx

# Define the cloud API URL
CLOUD_API_URL = "http://129.213.84.3:8002/store"

def read_battery_voltage():
    return 32

def read_battery_current():
    return 21

def read_battery_temp():
    return 30

def read_panel_voltage():
    return 4

def read_panel_current():
    return 5

def read_panel_temp():
    return 200

def read_panel_light():
    return 7

def read_load_current():
    return 69

def read_load_voltage():
    return 37

def car_counter():
    return 69

def truck_counter():
    return 32

def bike_counter():
    return 5

def ped_counter():
    return 16

def battery_percent():
    return 100

async def store_data_to_cloud(ccount: float, tcount: float, bcount: float, pcount: float, bpercent: float):
    data = {
        "car_count": ccount,
        "truck_count": tcount,
        "bike_count": bcount,
        "ped_count": pcount,
        "battery_percentage": bpercent
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(CLOUD_API_URL, json=data)
            response.raise_for_status()  # Raise an exception for HTTP errors
            print(f"Data stored successfully. Response: {response.text}")
    except httpx.RequestError as e:
        print(f"An error occurred while trying to reach the API: {e}")
    except httpx.HTTPStatusError as e:
        print(f"Error response {e.response.status_code}: {e.response.text}")

async def main():
    battery_current_reading = read_battery_current()
    battery_voltage_reading = read_battery_voltage()
    battery_temp_reading = read_battery_temp()
    panel_current_reading = read_panel_current()
    panel_voltage_reading = read_panel_voltage()
    panel_temp_reading = read_panel_temp()
    panel_light_reading = read_panel_light()
    load_current_reading = read_load_current()
    load_voltage_reading = read_load_voltage()

    car_count = car_counter()
    truck_count = truck_counter()
    bike_count = bike_counter()
    ped_count = ped_counter()
    battery_percentage = battery_percent()

    await store_data_to_cloud(car_count, truck_count, bike_count, ped_count, battery_percentage)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
