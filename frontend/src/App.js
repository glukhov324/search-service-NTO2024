import React, { useState } from 'react';
import SearchByText from './components/SearchByText';
import SearchByImage from './components/SearchByImage';
import MapResults from './components/MapResults';

function App() {
  const [results, setResults] = useState([]);

  const handleSearchResults = (data) => {
    let places = [];

    if (data.result) {
      places = data.result.map(item => ({
        sight_name: item.sight_name,
        lat: item.lat,
        lon: item.lon
      }));
    } else if (data.names_coords) {
      places = data.names_coords.map(item => ({
        sight_name: item.sight_name,
        lat: item.lat,
        lon: item.lon
      }));
    }

    setResults(places);
  };

  return (
    <div style={{
      maxWidth: '1000px',
      margin: '0 auto',
      fontFamily: 'Arial, sans-serif',
      padding: '20px'
    }}>
      <h1 style={{ textAlign: 'center', marginBottom: '30px' }}>🔍 Поиск достопримечательностей</h1>

      <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 45%', minWidth: '300px' }}>
          <SearchByText onResults={handleSearchResults} />
        </div>
        <div style={{ flex: '1 1 45%', minWidth: '300px' }}>
          <SearchByImage onResults={handleSearchResults} />
        </div>
      </div>

      {results.length > 0 && (
        <div style={{ marginTop: '40px' }}>
          <h2>📍 Найденные места на карте:</h2>
          <MapResults locations={results} />
        </div>
      )}
    </div>
  );
}

export default App;