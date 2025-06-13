import { useState } from "react";
import "./App.css"; // for styling

function App() {
  const [formData, setFormData] = useState({
    room_type: 0,
    longitude: -73.98615,
    host_listings_count: 1.0,
    accommodates: 1,
    host_acceptance_rate: 100.0,
    neighbourhood_cleansed: 2,
    amenities: 17182,
    instant_bookable: 1,
    beds: 1,
    bathrooms: 1,
  });

  const [predictedPrice, setPredictedPrice] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    let updatedValue = type === "checkbox" ? (checked ? 1 : 0) : value;

    // Update longitude when neighborhood changes
    if (name === "neighbourhood_cleansed") {
      const boroughLongitudes = {
        0: -73.8648, // Bronx
        1: -73.9442, // Brooklyn
        2: -73.98615, // Manhattan
        3: -73.7949, // Queens
        4: -74.1502, // Staten Island
      };
      updatedValue = parseInt(updatedValue);
      setFormData((prev) => ({
        ...prev,
        [name]: updatedValue,
        longitude: boroughLongitudes[updatedValue],
      }));
    } else {
      setFormData((prev) => ({
        ...prev,
        [name]: type === "number" ? parseFloat(updatedValue) : updatedValue,
      }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setPredictedPrice(null);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (response.ok) {
        setPredictedPrice(data.predicted_price);
      } else {
        setError(data.error || "Something went wrong");
      }
    } catch (err) {
      setError("Server error. Make sure backend is running.");
    }

    setLoading(false);
  };

  return (
    <div className="container">
      <h1>Airbnb Price Predictor</h1>
      <form onSubmit={handleSubmit} className="form">
        <label>
          Room Type:
          <select
            name="room_type"
            value={formData.room_type}
            onChange={handleChange}
          >
            <option value={0}>Entire home/apt</option>
            <option value={1}>Hotel room</option>
            <option value={2}>Private room</option>
            <option value={3}>Shared room</option>
          </select>
        </label>

        <label>
          Accommodates:
          <input
            type="number"
            name="accommodates"
            min="1"
            max="30"
            value={formData.accommodates}
            onChange={handleChange}
          />
        </label>

        <label>
          Neighbourhood:
          <select
            name="neighbourhood_cleansed"
            value={formData.neighbourhood_cleansed}
            onChange={handleChange}
          >
            <option value={0}>Bronx</option>
            <option value={1}>Brooklyn</option>
            <option value={2}>Manhattan</option>
            <option value={3}>Queens</option>
            <option value={4}>Staten Island</option>
          </select>
        </label>

        <label>
          Instant Bookable:
          <input
            type="checkbox"
            name="instant_bookable"
            checked={formData.instant_bookable === 1}
            onChange={handleChange}
          />
        </label>

        <label>
          Beds:
          <input
            type="number"
            name="beds"
            min="1"
            max="30"
            value={formData.beds}
            onChange={handleChange}
          />
        </label>

        <label>
          Bathrooms:
          <input
            type="number"
            name="bathrooms"
            min="1"
            max="30"
            value={formData.bathrooms}
            onChange={handleChange}
          />
        </label>

        <button type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Get Price Prediction"}
        </button>
      </form>

      {predictedPrice && (
        <div className="result">
          <h2>Predicted Price:</h2>
          <p>{predictedPrice}</p>
        </div>
      )}

      {error && <p className="error">{error}</p>}
    </div>
  );
}

export default App;
