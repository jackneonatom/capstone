
    async function fetchData() {
        try {const response = await fetch('http://pop-os.local:8000/count', {
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
            document.getElementById('car_count').innerHTML = `<span class="currentnumber1">${data.Car_Count || 'N/A'} </span>`;
            document.getElementById('truck_count').innerHTML = `<span class="currentnumber2">${data.Truck_Count || 'N/A'} </span>`;
            document.getElementById('bike_count').innerHTML = `<span class="currentnumber3">${data.Bike_Count || 'N/A'} </span>`;
            document.getElementById('ped_count').innerHTML = `<span class="currentnumber4">${data.Ped_Count || 'N/A'} </span>`;
           
           
            const currentNumber1Elements = document.querySelectorAll('.currentnumber1');
            currentNumber1Elements.forEach(element => {
                const currentValue = parseFloat(element.textContent.trim()) || 0;
                if (currentValue > 10) {
                    element.style.color = 'red';
                } else if (currentValue > 7) {
                    element.style.color = 'orange';
                } else {
                    element.style.color = 'blue';
                }
            });
            const currentNumber2Elements = document.querySelectorAll('.currentnumber2');
            currentNumber2Elements.forEach(element => {
                const currentValue = parseFloat(element.textContent.trim()) || 0;
                if (currentValue > 13) {
                    element.style.color = 'red';
                } else if (currentValue > 12) {
                    element.style.color = 'orange';
                } else {
                    element.style.color = 'blue';
                }
            });
            const currentNumber3Elements = document.querySelectorAll('.currentnumber3');
            currentNumber3Elements.forEach(element => {
                const currentValue = parseFloat(element.textContent.trim()) || 0;
                if (currentValue > 30) {
                    element.style.color = 'red';
                } else if (currentValue > 20) {
                    element.style.color = 'orange';
                } else {
                    element.style.color = 'blue';
                }
            });
            const currentNumber4Elements = document.querySelectorAll('.currentnumber4');
            currentNumber4Elements.forEach(element => {
                const currentValue = parseFloat(element.textContent.trim()) || 0;
                if (currentValue > 30) {
                    element.style.color = 'red';
                } else if (currentValue > 20) {
                    element.style.color = 'orange';
                } else {
                    element.style.color = 'blue';
                }
            });
           

            // // Fetch the LED status
            // const ledResponse = await fetch('http://pop-os.local:8000/led-status', {
            //     method: 'GET',
            //     headers: {
            //         'Content-Type': 'application/json',
            //     }
            // });
            // if (!ledResponse.ok) {
            //     throw new Error('Failed to fetch LED status');
            // }
            // const ledData = await ledResponse.json();
            // document.getElementById('ledStatus').innerHTML = `LED Status: ${ledData.status ? 'ON' : 'OFF'}`;

        } catch (error) {
            console.error('Error:', error);
        }
    }
    setInterval(fetchData, 500);
    // fetchHistoricalData();
    // fetchLedStatus();