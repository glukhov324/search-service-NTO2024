import React, { useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

// Исправление иконок Leaflet
import L from 'leaflet';
delete L.Icon.Default.prototype._getIconUrl;

L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet @1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet @1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet @1.9.4/dist/images/marker-shadow.png',
});

const MapResults = ({ locations }) => {
  const mapRef = useRef(null);
  const defaultCenter = locations[0] ? [locations[0].lat, locations[0].lon] : [55.7558, 37.6176];

  return (
    <MapContainer
      center={defaultCenter}
      zoom={10}
      style={{ height: '500px', width: '100%' }}
      ref={mapRef}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; OpenStreetMap contributors'
      />
      {locations.map((loc, idx) => (
        <Marker key={idx} position={[loc.lat, loc.lon]}>
          <Popup>{loc.sight_name}</Popup>
        </Marker>
      ))}
    </MapContainer>
  );
};

export default MapResults;