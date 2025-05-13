async function fetchData() {
    try {
        const response = await fetch('http://ec2-13-59-11-204.us-east-2.compute.amazonaws.com:8002/count', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        if (!response.ok) {
            throw new Error('Failed to fetch data');
        }
        const data = await response.json();
        console.log(data); // Log the data to verify its structure
        
        document.getElementById('car_count').innerHTML = `<span class="currentnumber1">${data.Car_Count || 'N/A'}</span>`;
        document.getElementById('truck_count').innerHTML = `<span class="currentnumber2">${data.Truck_Count || 'N/A'}</span>`;
        document.getElementById('bike_count').innerHTML = `<span class="currentnumber3">${data.Bike_Count || 'N/A'}</span>`;
        document.getElementById('ped_count').innerHTML = `<span class="currentnumber4">${data.Ped_Count || 'N/A'}</span>`;

        // Apply conditional formatting for vehicle counts
        updateElementColor('.currentnumber1', data.Car_Count, 10, 7);
        updateElementColor('.currentnumber2', data.Truck_Count, 13, 12);
        updateElementColor('.currentnumber3', data.Bike_Count, 30, 20);
        updateElementColor('.currentnumber4', data.Ped_Count, 30, 20);

    } catch (error) {
        console.error('Error:', error);
    }
}

function updateElementColor(selector, value, redThreshold, orangeThreshold) {
    const elements = document.querySelectorAll(selector);
    elements.forEach(element => {
        const currentValue = parseFloat(value) || 0;
        if (currentValue > redThreshold) {
            element.style.color = 'red';
        } else if (currentValue > orangeThreshold) {
            element.style.color = 'orange';
        } else {
            element.style.color = 'blue';
        }
    });
}

setInterval(fetchData, 500);
