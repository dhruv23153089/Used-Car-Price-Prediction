import { useState } from "react";
import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const CURRENT_YEAR = new Date().getFullYear();

const initialFormData = {
  brand: "",
  model_year: "",
  milage: "",
  fuel_type: "Gasoline",
  engine: "",
  transmission: "Automatic",
  accident: "None reported",
  clean_title: "Yes",
};

function CarForm() {
  const [formData, setFormData] = useState(initialFormData);
  const [price, setPrice] = useState(null);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData((current) => ({
      ...current,
      [name]: value,
    }));
  };

  const buildPayload = () => ({
    brand: formData.brand.trim(),
    model_year: Number(formData.model_year),
    milage: Number(formData.milage),
    fuel_type: formData.fuel_type,
    engine: formData.engine.trim(),
    transmission: formData.transmission,
    accident: formData.accident,
    clean_title: formData.clean_title,
  });

  const validateForm = () => {
    if (!formData.brand.trim()) {
      return "Brand is required.";
    }
    if (!formData.engine.trim()) {
      return "Engine description is required.";
    }
    if (!formData.model_year || Number(formData.model_year) < 1980 || Number(formData.model_year) > CURRENT_YEAR) {
      return `Model year must be between 1980 and ${CURRENT_YEAR}.`;
    }
    if (formData.milage === "" || Number(formData.milage) < 0) {
      return "Mileage must be a valid positive number.";
    }
    return "";
  };

  const predictPrice = async () => {
    const validationMessage = validateForm();
    if (validationMessage) {
      setError(validationMessage);
      setPrice(null);
      return;
    }

    setIsSubmitting(true);
    setError("");

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, buildPayload());
      setPrice(response.data.predicted_price);
    } catch (requestError) {
      const apiMessage = requestError.response?.data?.detail;
      setPrice(null);
      setError(typeof apiMessage === "string" ? apiMessage : "Prediction failed. Check that the backend is running.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="form-wrapper">
      <div className="form-grid">
        <label className="field">
          <span>Brand</span>
          <input
            type="text"
            name="brand"
            placeholder="Toyota"
            value={formData.brand}
            onChange={handleChange}
          />
        </label>

        <label className="field">
          <span>Model Year</span>
          <input
            type="number"
            name="model_year"
            placeholder="2020"
            min="1980"
            max={CURRENT_YEAR}
            value={formData.model_year}
            onChange={handleChange}
          />
        </label>

        <label className="field">
          <span>Mileage</span>
          <input
            type="number"
            name="milage"
            placeholder="35000"
            min="0"
            value={formData.milage}
            onChange={handleChange}
          />
        </label>

        <label className="field">
          <span>Fuel Type</span>
          <select name="fuel_type" value={formData.fuel_type} onChange={handleChange}>
            <option value="Gasoline">Gasoline</option>
            <option value="Hybrid">Hybrid</option>
            <option value="Diesel">Diesel</option>
            <option value="Electric">Electric</option>
            <option value="Plug-In Hybrid">Plug-In Hybrid</option>
            <option value="E85 Flex Fuel">E85 Flex Fuel</option>
            <option value="Unknown">Unknown</option>
          </select>
        </label>

        <label className="field field-wide">
          <span>Engine</span>
          <input
            type="text"
            name="engine"
            placeholder="300HP 3.5L V6"
            value={formData.engine}
            onChange={handleChange}
          />
        </label>

        <label className="field">
          <span>Transmission</span>
          <select name="transmission" value={formData.transmission} onChange={handleChange}>
            <option value="Automatic">Automatic</option>
            <option value="CVT">CVT</option>
            <option value="Manual">Manual</option>
          </select>
        </label>

        <label className="field">
          <span>Accident History</span>
          <select name="accident" value={formData.accident} onChange={handleChange}>
            <option value="None reported">No accident reported</option>
            <option value="At least 1 accident or damage reported">Had accident or damage</option>
          </select>
        </label>

        <label className="field">
          <span>Clean Title</span>
          <select name="clean_title" value={formData.clean_title} onChange={handleChange}>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
            <option value="Unknown">Unknown</option>
          </select>
        </label>
      </div>

      <button className="predict-button" onClick={predictPrice} disabled={isSubmitting}>
        {isSubmitting ? "Calculating estimate..." : "Predict Price"}
      </button>

      {error && <p className="error-message">{error}</p>}

      {price !== null && (
        <div className="result-card">
          <p>Estimated Price</p>
          <h3>${new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(price)}</h3>
        </div>
      )}
    </div>
  );
}

export default CarForm;
