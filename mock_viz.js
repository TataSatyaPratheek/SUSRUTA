import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { PieChart, Pie, Cell } from 'recharts';
import { LineChart, Line } from 'recharts';
import { AlertTriangle, ThumbsUp, Clock, Activity, Calendar, Users, FileText } from 'lucide-react';

// Mock data for the treatment recommendation system
const patientData = {
  id: "P-0023",
  age: 56,
  gender: "Female",
  diagnosis: "Glioblastoma, IDH-wildtype",
  karnofsky: 90,
  tumorLocation: "Right temporal",
  priorTreatments: ["Surgical resection (GTR)", "Radiation (60Gy/30)"]
};

const treatmentOptions = [
  {
    id: 1,
    name: "Temozolomide (Standard)",
    category: "Chemotherapy",
    responseProb: 0.72,
    survivalDays: 485,
    survivalCI: [410, 560],
    toxicityRisk: 0.15,
    uncertainty: 0.12
  },
  {
    id: 2,
    name: "Bevacizumab + Irinotecan",
    category: "Combination therapy",
    responseProb: 0.61,
    survivalDays: 380,
    survivalCI: [290, 470],
    toxicityRisk: 0.33,
    uncertainty: 0.18
  },
  {
    id: 3,
    name: "TTFields + Temozolomide",
    category: "Combination therapy",
    responseProb: 0.78,
    survivalDays: 520,
    survivalCI: [430, 610],
    toxicityRisk: 0.22,
    uncertainty: 0.14
  }
];

const featureImportance = [
  { name: "Tumor volume", value: 0.23 },
  { name: "IDH status", value: 0.19 },
  { name: "Patient age", value: 0.16 },
  { name: "MGMT status", value: 0.14 },
  { name: "Karnofsky score", value: 0.11 },
  { name: "Extent of resection", value: 0.09 },
  { name: "Prior radiation", value: 0.08 }
];

const survivalCurveData = Array(24).fill().map((_, i) => {
  const month = i + 1;
  return {
    month,
    treatment1: Math.exp(-month / 18) * 100,
    treatment2: Math.exp(-month / 14) * 100,
    treatment3: Math.exp(-month / 20) * 100,
  };
});

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658'];

const TreatmentDashboard = () => {
  const [selectedTreatment, setSelectedTreatment] = useState(treatmentOptions[0]);

  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-indigo-700 px-6 py-4 text-white">
        <h1 className="text-2xl font-bold">Glioma Treatment Recommendation System</h1>
      </div>
      
      {/* Main content */}
      <div className="flex flex-col md:flex-row p-4 gap-4">
        {/* Left sidebar - Patient info */}
        <div className="w-full md:w-1/4 bg-white rounded-lg shadow p-4">
          <h2 className="text-xl font-semibold mb-4 text-gray-800">Patient Information</h2>
          <div className="space-y-2">
            <div className="flex items-center">
              <Users className="h-5 w-5 mr-2 text-indigo-600" />
              <span className="font-medium">ID:</span>
              <span className="ml-2">{patientData.id}</span>
            </div>
            <div className="flex items-center">
              <Calendar className="h-5 w-5 mr-2 text-indigo-600" />
              <span className="font-medium">Age:</span>
              <span className="ml-2">{patientData.age}</span>
            </div>
            <div className="flex items-center">
              <Activity className="h-5 w-5 mr-2 text-indigo-600" />
              <span className="font-medium">Diagnosis:</span>
              <span className="ml-2">{patientData.diagnosis}</span>
            </div>
            <div className="flex items-center">
              <FileText className="h-5 w-5 mr-2 text-indigo-600" />
              <span className="font-medium">Location:</span>
              <span className="ml-2">{patientData.tumorLocation}</span>
            </div>
          </div>
          
          <h3 className="text-lg font-semibold mt-6 mb-2 text-gray-800">Prior Treatments</h3>
          <ul className="list-disc pl-6 text-gray-700">
            {patientData.priorTreatments.map((treatment, index) => (
              <li key={index}>{treatment}</li>
            ))}
          </ul>
          
          <h3 className="text-lg font-semibold mt-6 mb-2 text-gray-800">Key Predictors</h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart 
                layout="vertical" 
                data={featureImportance}
                margin={{ top: 5, right: 5, bottom: 5, left: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 0.3]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={100} />
                <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} />
                <Bar dataKey="value" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        {/* Main content - Treatment recommendations */}
        <div className="w-full md:w-3/4 space-y-4">
          {/* Treatment options */}
          <div className="bg-white rounded-lg shadow p-4">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Recommended Treatments</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {treatmentOptions.map((treatment) => (
                <div 
                  key={treatment.id}
                  className={`border rounded-lg p-4 cursor-pointer transition-all ${selectedTreatment.id === treatment.id ? 'border-indigo-500 bg-indigo-50 shadow-md' : 'border-gray-200 hover:border-indigo-300'}`}
                  onClick={() => setSelectedTreatment(treatment)}
                >
                  <h3 className="font-semibold text-lg">{treatment.name}</h3>
                  <p className="text-gray-500 text-sm">{treatment.category}</p>
                  
                  <div className="mt-3 space-y-2">
                    <div className="flex items-center">
                      <ThumbsUp className="h-4 w-4 mr-2 text-green-600" />
                      <span className="text-sm">Response: <span className="font-medium">{(treatment.responseProb * 100).toFixed(0)}%</span></span>
                    </div>
                    <div className="flex items-center">
                      <Clock className="h-4 w-4 mr-2 text-blue-600" />
                      <span className="text-sm">Survival: <span className="font-medium">{treatment.survivalDays} days</span></span>
                    </div>
                    <div className="flex items-center">
                      <AlertTriangle className="h-4 w-4 mr-2 text-amber-600" />
                      <span className="text-sm">Toxicity: <span className="font-medium">{(treatment.toxicityRisk * 100).toFixed(0)}%</span></span>
                    </div>
                  </div>
                  
                  {/* Uncertainty indicator */}
                  <div className="mt-3 flex items-center">
                    <div className="h-2 flex-grow rounded-full bg-gray-200">
                      <div 
                        className="h-2 rounded-full bg-indigo-600" 
                        style={{ width: `${(1 - treatment.uncertainty) * 100}%` }}
                      ></div>
                    </div>
                    <span className="ml-2 text-xs text-gray-500">
                      {treatment.uncertainty < 0.15 ? 'High confidence' : 'Moderate confidence'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          {/* Selected treatment details */}
          <div className="bg-white rounded-lg shadow p-4">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">
              Treatment Details: {selectedTreatment.name}
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Survival curve */}
              <div>
                <h3 className="text-lg font-medium mb-2">Projected Survival</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={survivalCurveData} margin={{ top: 5, right: 5, bottom: 20, left: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" label={{ value: 'Months', position: 'insideBottom', offset: -15 }} />
                      <YAxis domain={[0, 100]} label={{ value: 'Survival (%)', angle: -90, position: 'insideLeft' }} />
                      <Tooltip />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey={`treatment${selectedTreatment.id}`} 
                        name={selectedTreatment.name}
                        stroke="#8884d8" 
                        strokeWidth={3} 
                        dot={false} 
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              
              {/* Risk/benefit chart */}
              <div>
                <h3 className="text-lg font-medium mb-2">Risk/Benefit Analysis</h3>
                <div className="h-64 flex items-center justify-center">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={[
                          { name: 'Response Benefit', value: selectedTreatment.responseProb },
                          { name: 'Toxicity Risk', value: selectedTreatment.toxicityRisk },
                          { name: 'Uncertainty', value: selectedTreatment.uncertainty }
                        ]}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={80}
                        paddingAngle={5}
                        dataKey="value"
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                        labelLine={false}
                      >
                        <Cell fill="#4CAF50" />
                        <Cell fill="#FF5722" />
                        <Cell fill="#9E9E9E" />
                      </Pie>
                      <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
            
            {/* Explanation of recommendation */}
            <div className="mt-6 p-4 bg-indigo-50 rounded-lg">
              <h3 className="text-lg font-medium mb-2">Why This Treatment?</h3>
              <p className="text-gray-700">
                {selectedTreatment.name} is recommended based on the patient's favorable {selectedTreatment.id === 1 ? 'MGMT methylation status' : selectedTreatment.id === 2 ? 'vascular characteristics' : 'KPS score and tumor location'}. 
                The model predicts a {(selectedTreatment.responseProb * 100).toFixed(0)}% chance of positive response,
                with an estimated median survival of {selectedTreatment.survivalDays} days 
                (95% CI: {selectedTreatment.survivalCI[0]}-{selectedTreatment.survivalCI[1]} days).
              </p>
              
              <div className="mt-3 flex items-center">
                <div className="w-4 h-4 bg-amber-500 rounded-full mr-2"></div>
                <p className="text-sm text-gray-600">
                  <strong>Consider:</strong> Monitor for {selectedTreatment.id === 1 ? 'myelosuppression' : selectedTreatment.id === 2 ? 'hypertension and proteinuria' : 'skin irritation at device contact sites'}.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TreatmentDashboard;