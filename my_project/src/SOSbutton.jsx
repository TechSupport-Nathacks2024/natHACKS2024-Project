import React, { useState, useEffect } from 'react';
import './SOSbutton.css';

const SOSButton = () => {
  const emergencyNumber = 'tel:+1234567890'; // Replace with actual emergency number

  const handleDragStart = (event) => {
    event.dataTransfer.setData('text/plain', emergencyNumber);
  };

  const handleDragEnd = () => {
    // Initiate the call when the button is dragged
    window.location.href = emergencyNumber;
  };

  return (
    <div
      id="sosButton"
      draggable="true"
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
      style={styles.sosButton}
    >
      <div style={styles.circle}>SOS</div>
      <div style={styles.text}>Emergency Call</div>
    </div>
  );
};

const styles = {
  sosButton: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '10px 20px',
    backgroundColor: '#ff0000',
    borderRadius: '50px',
    color: 'white',
    fontWeight: 'bold',
    cursor: 'grab',
    userSelect: 'none',
  },
  circle: {
    fontSize: '25px',
  },
  text: {
    marginLeft: '10px',
    fontSize: '25px',
  },
};

export default SOSButton;

