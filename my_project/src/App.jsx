import React, { useState, useEffect } from 'react';
import './index.css';
import './App.css';
import './Navigation-bar.css';
import ConnectButton from './ConnectButton.jsx';
import toggle_dark from "./assets/night.png";
import toggle_light from './assets/day.png';
import SOSButton from './SOSbutton.jsx';
import Weather1 from './Weather.jsx';
import fileContent from './alerts_log.txt?raw'
// import Cam from './cam.jsx';

// Navigationbar component
function Navigationbar({ theme, setTheme }) {
  const toggleThemeMode = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  return (
    <div className="navbar">
      <h1 className={`bullet ${theme === 'light' ? 'text-#EEEEEE;' : 'text-#333'}`}>
        Drive Safe
      </h1>
      <img 
        src={theme === 'light' ? toggle_dark : toggle_light} 
        alt="toggle icon" 
        onClick={toggleThemeMode} 
        className="toggle-icon" 
      />
    </div>
  );
}

// Popup Component
function Popup({ onClose, children }) {
  return (
    <div className="popup-overlay" onClick={onClose}>
      <div className="popup-content" onClick={(e) => e.stopPropagation()}>
        {children}
        <button className="close-btn" onClick={onClose}>Close</button>
      </div>
    </div>
  );
}

// Main App component
function App() {
  const [theme, setTheme] = useState('light');
  const [isCameraPopupVisible, setIsCameraPopupVisible] = useState(false);
  const [isHistoricPopupVisible, setIsHistoricPopupVisible] = useState(false);

  useEffect(() => {
    document.body.className = theme === 'light' ? 'light-mode' : 'dark-mode';
  }, [theme]);

  // Show camera popup
  const handleCameraConnect = () => {
    setIsCameraPopupVisible(true);
  };

  // Show historic data popup
  const handleHistoricDataClick = () => {
    setIsHistoricPopupVisible(true);
  };

  // Close all popups
  const closePopup = () => {
    setIsCameraPopupVisible(false);
    setIsHistoricPopupVisible(false);
  };

  return (
    <div className="app-container">
      <div className="Weather-container">
        <Weather1 />
      </div>
      <div className="SOS">
        <SOSButton />
      </div>
      <Navigationbar theme={theme} setTheme={setTheme} />

      <div className="button-container">
        <ConnectButton 
          theme={theme} 
          onCameraConnect={handleCameraConnect} 
          onHistoricDataClick={handleHistoricDataClick} 
        />
      </div>

      {/* Camera Popup */}
      {isCameraPopupVisible && (
        <Popup onClose={closePopup}>
          <div className='cam-cont'>
            {/* <Cam/> */}
          <p className='camera1'><img
        className="image"
        src="http://localhost:7000/video_feed"
        alt="Video"
      />
      <div className="alert-box">
        <pre>{fileContent}</pre> {/* Display the alert message */}
      </div></p>
          </div>
        </Popup>
      )}
      
      {/* Historic Data Popup */}
      {isHistoricPopupVisible && (
        <Popup onClose={closePopup}>
          <h2>Historic Data</h2>
          <p className='historic-para'>Here’s where the historic data will be displayed.</p>
        </Popup>
      )}
    </div>
  );
}

export default App;
