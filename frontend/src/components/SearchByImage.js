import React, { useState } from 'react';
import axios from 'axios';

const CITIES = ["Екатеринбург", "Нижний Новгород", "Владимир", "Ярославль"];

const SearchByImage = ({ onResults }) => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [city, setCity] = useState(CITIES[0]);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    if (selectedFile) {
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target.result);
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('image_file', file);
    formData.append('city', city);

    try {
      const response = await axios.post('http://localhost:8000/search/by_image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      if (typeof onResults === 'function') {
        onResults(response.data);
      }
    } catch (error) {
      console.error('Ошибка при поиске по изображению:', error);
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
      <h2>📷 Поиск по изображению</h2>

      <div style={{ marginBottom: '10px' }}>
        <label htmlFor="city-select-img">Выберите город:</label>
        <select
          id="city-select-img"
          value={city}
          onChange={(e) => setCity(e.target.value)}
          style={{ marginLeft: '10px', padding: '5px' }}
        >
          {CITIES.map((cityName, idx) => (
            <option key={idx} value={cityName}>{cityName}</option>
          ))}
        </select>
      </div>

      <input type="file" accept="image/*" onChange={handleFileChange} />
      {preview && (
        <img
          src={preview}
          alt="Превью"
          style={{ width: '300px', margin: '10px 0', borderRadius: '8px' }}
        />
      )}

      <button
        onClick={handleUpload}
        disabled={loading || !file}
        style={{
          marginTop: '10px',
          padding: '10px 20px',
          fontSize: '16px',
          cursor: loading ? 'not-allowed' : 'pointer'
        }}
      >
        {loading ? 'Анализ...' : 'Загрузить и проанализировать'}
      </button>
    </div>
  );
};

export default SearchByImage;