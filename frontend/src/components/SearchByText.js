import React, { useState } from 'react';
import axios from 'axios';

const CITIES = ["Екатеринбург", "Нижний Новгород", "Владимир", "Ярославль"];

const SearchByText = ({ onResults }) => {
  const [query, setQuery] = useState('');
  const [city, setCity] = useState(CITIES[0]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);

    const formData = new FormData();
    formData.append('text', query);
    formData.append('city', city);

    try {
      const response = await axios.post('http://localhost:8000/search/by_text', formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      });

      if (typeof onResults === 'function') {
        onResults(response.data);
      }
    } catch (error) {
      console.error('Ошибка при поиске по тексту:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      padding: '20px',
      border: '1px solid #ddd',
      borderRadius: '8px',
      backgroundColor: '#f9f9f9'
    }}>
      <h2>🔍 Поиск по тексту</h2>

      <div style={{ marginBottom: '10px' }}>
        <label htmlFor="city-select">Выберите город:</label>
        <select
          id="city-select"
          value={city}
          onChange={(e) => setCity(e.target.value)}
          style={{ marginLeft: '10px', padding: '5px' }}
        >
          {CITIES.map((cityName, idx) => (
            <option key={idx} value={cityName}>
              {cityName}
            </option>
          ))}
        </select>
      </div>

      <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
        <input
          type="text"
          placeholder="Введите название или описание..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ flex: 1, padding: '10px', fontSize: '16px' }}
        />
        <button
          onClick={handleSearch}
          disabled={loading}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            cursor: loading ? 'not-allowed' : 'pointer'
          }}
        >
          {loading ? 'Поиск...' : 'Найти'}
        </button>
      </div>
    </div>
  );
};

export default SearchByText;