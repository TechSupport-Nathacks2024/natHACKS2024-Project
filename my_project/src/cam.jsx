import React, { useEffect, useState } from "react";
import "./Cam.css"; // Import the CSS file
import fileContent from './alerts_log.txt?raw';
import "./cam1.css"

const Cam = () => {
  const [alert, setAlert] = useState("Loading..."); // State to store the alert message

  // Function to fetch the last alert
  const fetchLastAlert = async () => {
    try {
      const response = await fetch("http://localhost:7000/get_last_alert");
      const data = await response.json();
      setAlert(data.last_alert);
    } catch (error) {
      console.error("Error fetching last alert:", error);
      setAlert("Unable to fetch alerts.");
    }
  };

  useEffect(() => {
    fetchLastAlert(); // Fetch the last alert on component mount
    const interval = setInterval(fetchLastAlert, 5000); // Update every 5 seconds
    return () => clearInterval(interval); // Cleanup on unmount
  }, []);

  return (
    <div>
      <img
        className="image"
        src="http://localhost:7000/video_feed"
        alt="Video"
      />
      <div className="alert-box">
        <pre>{fileContent}</pre> {/* Display the alert message */}
      </div>
    </div>
  );
};

export default Cam;
