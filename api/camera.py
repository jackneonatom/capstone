import asyncio
from vehcnt import store_data  # Assuming store_data is defined in app.py

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

    # Call the updated store_data function
    await store_data(car_count, truck_count, bike_count, ped_count, battery_percentage)
    print("Data stored successfully.")

asyncio.run(main())
