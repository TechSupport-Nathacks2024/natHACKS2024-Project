import React, { useState, useEffect, useRef } from 'react';
import './Weather.css';
import search_ic from './assets/search.png';
import clear from './assets/clear.png';
import humid from './assets/humidity.png';
import cloud from './assets/cloud.png';
import drizzle from './assets/drizzle.png';
import rain from './assets/rain.png';
import snow from './assets/snow.png';
import wind from './assets/wind.png';

const Weather1 = () => {
    const inputRef = useRef();
    const [weatherData, setWeatherData] = useState(false);
    const [greetingMessage, setGreetingMessage] = useState('');
    
    const allIcons = {
        "01d": clear,
        "01n": clear,
        "02d": cloud,
        "02n": cloud,
        "03d": cloud,
        "03n": cloud,
        "04d": drizzle,
        "04n": drizzle,
        "09d": rain,
        "09n": rain,
        "10d": rain,
        "10n": rain,
        "13d": snow,
        "13n": snow,
    };

    // Time-based greeting
    const getTimeBasedGreeting = () => {
        const hours = new Date().getHours();
        if (hours >= 5 && hours < 12) return "Good Morning";
        else if (hours >= 12 && hours < 18) return "Good Afternoon";
        else return "Good Evening";
    };

    // Weather-based message
    const generateWeatherMessage = (weather, humidity, windSpeed) => {
        const greeting = getTimeBasedGreeting();
        
        if (weather.includes("clear")) {
            return `${greeting}, it's sunny today, such a good day to drive!`;
        } else if (weather.includes("clouds")) {
            return `${greeting}, it's a bit cloudy, but a safe drive should be fine!`;
        } else if (weather.includes("rain")) {
            return humidity > 80 
                ? `${greeting}, it's raining with high humidity. Drive cautiously!` 
                : `${greeting}, light rain today, make sure your wipers work!`;
        } else if (weather.includes("snow")) {
            return windSpeed > 15 
                ? `${greeting}, it's snowing heavily with strong winds. Reconsider your drive today!` 
                : `${greeting}, it's snowing today, be cautious if you drive.`;
        } else if (weather.includes("wind")) {
            return windSpeed > 20 
                ? `${greeting}, it's windy and could be unsafe to drive.` 
                : `${greeting}, it's a bit windy, but should be manageable.`;
        } else {
            return `${greeting}, drive safely and check the weather as it may vary.`;
        }
    };

    // Search and update weather data
    const search = async (city) => {
        if (city === "") {
            alert("Enter city name");
            return;
        }
        try {
            const url = `https://api.openweathermap.org/data/2.5/weather?q=${city}&units=metric&appid=${import.meta.env.VITE_APP_ID}`;
            const response = await fetch(url);
            const data = await response.json();
            const icon = allIcons[data.weather[0].icon] || clear;

            const updatedWeatherData = {
                humidity: data.main.humidity,
                windSpeed: data.wind.speed,
                temperature: Math.floor(data.main.temp),
                location: data.name,
                icon: icon,
                weather: data.weather[0].main.toLowerCase()
            };
            setWeatherData(updatedWeatherData);

            // Generate the greeting message based on weather conditions
            const message = generateWeatherMessage(updatedWeatherData.weather, updatedWeatherData.humidity, updatedWeatherData.windSpeed);
            setGreetingMessage(message);

        } catch (error) {
            console.error("Error fetching weather data:", error);
        }
    };

    // Initial weather search for Edmonton on component load
    useEffect(() => {
        search("Edmonton");
    }, []);

    return (
        <div className='weather'>
            <div className='no-text'>
            <div className="search-bar">
                <input ref={inputRef} type="text" placeholder='Search' />
                <img src={search_ic} alt="search icon" onClick={() => search(inputRef.current.value)} />
            </div>
            <img src={weatherData.icon} alt="weather icon" className='weather-icon' />
            <p className="temp">{weatherData.temperature}°C</p>
            <p className="loc">{weatherData.location}</p>
            <div className="weather-data">
                <div className="col">
                    <img src={humid} alt="humidity icon" />
                    <div>
                        <p>{weatherData.humidity}%</p>
                        <span>Humidity</span>
                    </div>
                </div>
                <div className="col">
                    <img src={wind} alt="wind icon" />
                    <div>
                        <p>{weatherData.windSpeed}Km/h</p>
                        <span>Wind Speed</span>
                    </div>
                </div>
            </div>
            </div>
            <div className="greet">
            {/* Display the greeting message */}
            <p className="greeting-message">{greetingMessage}</p>
        </div>  
        </div>
    );
}

export default Weather1;
