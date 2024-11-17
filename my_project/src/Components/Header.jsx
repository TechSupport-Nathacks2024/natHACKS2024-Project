import {FaMoon, FaSun} from "react-icons/fa";
import '../index.css';

import React from 'react';


const ConnectButton = () => {
  return (
    <div className="flex justify-center items-center h-screen">
      <button className="bg-white text-white py-2 px-4 text-lg rounded-lg hover:bg-blue-700">
        Connect to Camera
      </button>
    </div>
  );
};

export default ConnectButton;

