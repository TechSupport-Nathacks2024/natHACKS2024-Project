import './index.css';
import './App.css';
import React, { useRef, useState } from 'react';

const ConnectButton = ({ theme, onCameraConnect, onHistoricDataClick }) => {
  const [isCameraOn, setIsCameraOn] = useState(false);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const handleCameraConnect = async () => {
    onCameraConnect(); // Show popup on camera connect
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      streamRef.current = stream;
      setIsCameraOn(true);
    } catch (error) {
      console.error("Camera access error:", error);
      alert("Failed to access the camera. Please check your permissions.");
    }
  };

  const handleCameraDisconnect = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
      setIsCameraOn(false);
    }
  };

  const handleMuseConnect = () => {
    alert("Connecting to Muse...");
  };

  return (
    <div className="button-container fixed right-5 flex flex-col grid-cols-10 grid-rows-10 mb-8 ml-auto pr-4">
      <button
        onClick={onHistoricDataClick} // Trigger historic data popup
        className="button1 text-white py-4 px-8 text-xl rounded-lg 
               bg-[#134B70] hover:bg-[#283a62] w-auto"
      >
        Historic Data
      </button>
      <button
        onClick={handleCameraConnect}
        className="button1 Historic text-white py-4 px-8 text-xl rounded-lg 
               bg-[#134B70] hover:bg-[#283a62] w-auto mb-4 transition-all duration-300 opacity-100"
      >
        Connect to Camera
      </button>
      <button
        onClick={handleMuseConnect}
        className="button1 text-white py-4 px-8 text-xl rounded-lg 
               bg-[#134B70] hover:bg-[#283a62] w-auto"
      >
        Connect to Muse
      </button>

      {isCameraOn && (
        <div className="flex flex-col items-center mt-4">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="w-64 h-48 border rounded-lg"
          />
          <button
            onClick={handleCameraDisconnect}
            className={`py-2 px-4 text-lg rounded-lg mt-4 ${theme === 'light' ? 'bg-blue-500 hover:bg-blue-700' : 'bg-blue-600 hover:bg-blue-800'}`}
          >
            Disconnect Camera
          </button>
        </div>
      )}
    </div>
  );
};

export default ConnectButton;
