import React from 'react';
import './NavBar.css';
import logo_light from '../../assets/logo-black.png';
import logo_dark from '../../assets/logo-white.png';
import search_icon from '../../assets/search-w.png';
import search_icon_dark from '../../assets/search-b.png';
import toggle_dark from '../../assets/night.png';
import toggle_light from '../../assets/day.png';

function Navigationbar({ theme, setTheme }) {
  const toggleThemeMode = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  return (
    <div className={`navbar ${theme}`}>
      <img src={theme === 'light' ? logo_light : logo_dark} alt="logo" className="logo" />
      <ul className={`bullet ${theme}`}>
        <li>Home</li>
        <li>Contact Us</li>
        <li id="clock">Time</li>
      </ul>
      <div className={`search-bar ${theme}`}>
        <input type="text" placeholder="Search..." />
        <img src={theme === 'light' ? search_icon : search_icon_dark} alt="search icon" />
      </div>
      <img
        src={theme === 'light' ? toggle_dark : toggle_light}
        alt="toggle icon"
        onClick={toggleThemeMode}
        className="toggle-icon"
      />
    </div>
  );
}

export default Navigationbar;
