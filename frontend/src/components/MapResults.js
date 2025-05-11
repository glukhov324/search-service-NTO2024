import React, { useRef, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

// Исправление иконок Leaflet
import L from 'leaflet';
delete L.Icon.Default.prototype._getIconUrl;

L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet @1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet @1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet @1.9.4/dist/images/marker-shadow.png',
});

// Компонент для управления центром карты без принудительного ресета
const MapController = ({ center }) => {
  const map = useMap();

  useEffect(() => {
    if (map && center) {
      map.setView(center, map.getZoom());
    }
  }, [center]);

  return null;
};

const MapResults = ({ locations, city }) => {
  const mapRef = useRef(null);
  const defaultCenter = locations[0] ? [locations[0].lat, locations[0].lon] : [55.7558, 37.6176];

  // Предотвращаем повторную инициализацию карты
  useEffect(() => {
    if (mapRef.current && mapRef.current._leaflet_id) {
      console.warn('Карта уже инициализирована');
    }
  }, []);

  return (
    <MapContainer
      ref={mapRef}
      center={defaultCenter}
      zoom={10}
      style={{ height: '500px', width: '100%' }}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; OpenStreetMap contributors'
      />
      <MapController center={defaultCenter} />

      {locations.map((loc, idx) => (
        <Marker key={idx} position={[loc.lat, loc.lon]}>
          <Popup>{loc.sight_name}</Popup>
        </Marker>
      ))}
    </MapContainer>
  );
};

export default MapResults;