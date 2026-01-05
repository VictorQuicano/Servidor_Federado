import React, { useState, useEffect } from 'react';
import { Clock, AlertCircle, CheckCircle, Users } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function TrainingSessionsDashboard() {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedSession, setSelectedSession] = useState(null);

  useEffect(() => {
    fetchSessions();
  }, []);

  const fetchSessions = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${import.meta.env.VITE_API_URL}/training/sessions`);
      if (!response.ok) throw new Error('Failed to fetch sessions');
      const data = await response.json();
      setSessions(data);
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching sessions:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString('es-ES', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const calculateDuration = (startTime, endTime) => {
    if (!endTime) return 'En progreso...';
    const start = new Date(startTime);
    const end = new Date(endTime);
    const diffMs = end - start;
    const diffMins = Math.floor(diffMs / 60000);
    const mins = diffMins % 60;
    const hours = Math.floor(diffMins / 60);
    
    if (hours > 0) return `${hours}h ${mins}m`;
    return `${mins}m`;
  };

  const getStatusIcon = (status) => {
    if (status === 'ACTIVE') {
      return <Clock className="w-5 h-5 text-blue-500" />;
    }
    return <AlertCircle className="w-5 h-5 text-orange-500" />;
  };

  const getStatusColor = (status) => {
    if (status === 'ACTIVE') return 'bg-blue-900/20 border-blue-800/50 hover:border-blue-500/50 shadow-blue-900/20';
    return 'bg-orange-900/20 border-orange-800/50 hover:border-orange-500/50 shadow-orange-900/20';
  };

  if (selectedSession) {
    return <SessionDetail session={selectedSession} onBack={() => setSelectedSession(null)} />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Sesiones de Entrenamiento</h1>
          <p className="text-gray-400">Gestiona y visualiza todas tus sesiones de entrenamiento</p>
        </div>

        {loading && (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mb-4"></div>
              <p className="text-gray-300">Cargando sesiones...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-900 border border-red-700 rounded-lg p-4 mb-6 text-red-200">
            <p className="font-semibold">Error al cargar sesiones</p>
            <p className="text-sm">{error}</p>
            <button 
              onClick={fetchSessions}
              className="mt-3 px-4 py-2 bg-red-700 hover:bg-red-600 rounded text-white text-sm font-medium transition"
            >
              Reintentar
            </button>
          </div>
        )}

        {!loading && !error && sessions.length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <p>No hay sesiones de entrenamiento disponibles</p>
          </div>
        )}

        {!loading && !error && sessions.length > 0 && (
          <div className="grid gap-4 md:grid-cols-1 lg:grid-cols-2">
            {sessions.map((session) => (
              <button
                key={session.id}
                onClick={() => setSelectedSession(session)}
                className={`text-left p-6 rounded-xl border transition-all hover:shadow-2xl cursor-pointer ${getStatusColor(session.status)} hover:scale-[1.02] transform backdrop-blur-sm`}
              >
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-bold text-white">
                      Sesión #{session.id}
                    </h3>
                    <p className="text-sm text-gray-400">{formatDate(session.start_time)}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    {getStatusIcon(session.status)}
                    <span className={`px-3 py-1 rounded-full text-xs font-bold border ${
                      session.status === 'ACTIVE' 
                        ? 'bg-blue-500/10 text-blue-400 border-blue-500/20' 
                        : 'bg-orange-500/10 text-orange-400 border-orange-500/20'
                    }`}>
                      {session.status}
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-800/40 p-3 rounded-lg border border-gray-700/30">
                    <p className="text-xs text-gray-400 mb-1">Duración</p>
                    <p className="font-semibold text-white">
                      {calculateDuration(session.start_time, session.end_time)}
                    </p>
                  </div>
                  <div className="bg-gray-800/40 p-3 rounded-lg border border-gray-700/30">
                    <p className="text-xs text-gray-400 mb-1">Rondas</p>
                    <p className="font-semibold text-white">{session.total_rounds}</p>
                  </div>
                </div>

                <p className="text-xs text-gray-500 mt-4 flex items-center gap-1 italic">
                  <span>Haz clic para ver detalles</span>
                  <span className="animate-pulse">→</span>
                </p>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function SessionClientsComponent({ sessionId }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (sessionId === null || sessionId === undefined) {
      setLoading(false);
      return;
    }

    fetchSessionClients();
  }, [sessionId]);

  const fetchSessionClients = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${import.meta.env.VITE_API_URL}/training/${sessionId}/details/`);
      if (!response.ok) throw new Error('Failed to fetch session clients');
      const result = await response.json();
      setData(result);
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching session clients:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="text-center py-8">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mb-2"></div>
        <p className="text-gray-300">Cargando detalles de clientes...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 text-red-200">
        <p className="font-semibold">Error al cargar datos</p>
        <p className="text-sm">{error}</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-center py-8 text-gray-400">
        <p>Esperando ID de sesión...</p>
      </div>
    );
  }

  const clients = data.clients || [];
  const globalMetrics = data.global_metrics || [];

  // Preparar datos para gráficos de métricas
  const metricsData = globalMetrics.map(metric => ({
    round: metric.round_number,
    avg_train_loss: parseFloat(metric.metrics.avg_train_loss.toFixed(6)),
    avg_actor_loss: parseFloat(metric.metrics.avg_actor_loss.toFixed(6)),
    avg_val_reward: parseFloat(metric.metrics.avg_val_reward.toFixed(6))
  })).sort((a, b) => a.round - b.round);

  const getStatusColor = (status) => {
    if (status === 'IDLE') return 'bg-gray-500/20 text-gray-300';
    if (status === 'TRAINING') return 'bg-blue-500/20 text-blue-300';
    if (status === 'READY') return 'bg-green-500/20 text-green-300';
    return 'bg-yellow-500/20 text-yellow-300';
  };

  return (
    <div className="space-y-8">
      {/* Sección de Clientes */}
      <div>
        <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
          <Users className="w-6 h-6" />
          Clientes Conectados ({clients.length})
        </h2>
        
        {clients.length === 0 ? (
          <p className="text-gray-400">No hay clientes conectados</p>
        ) : (
          <div className="grid gap-4 md:grid-cols-1 lg:grid-cols-2">
            {clients.map((client, idx) => (
              <div key={idx} className="bg-gray-700/50 border border-gray-600 rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <p className="text-sm text-gray-400">User ID</p>
                    <p className="text-white font-semibold">{client.user_id}</p>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${getStatusColor(client.current_status)}`}>
                    {client.current_status}
                  </span>
                </div>
                
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <p className="text-gray-400">Ronda actual</p>
                    <p className="text-white font-semibold">{client.current_round}</p>
                  </div>
                  <div>
                    <p className="text-gray-400">Último visto</p>
                    <p className="text-white text-xs">{new Date(client.last_seen).toLocaleTimeString('es-ES')}</p>
                  </div>
                </div>

                {client.ip_address && (
                  <div className="mt-3 pt-3 border-t border-gray-600">
                    <p className="text-gray-400 text-xs">IP Address</p>
                    <p className="text-white font-mono text-sm">{client.ip_address}</p>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Sección de Métricas Globales */}
      {metricsData.length > 0 && (
        <div>
          <h2 className="text-xl font-semibold text-white mb-6">Métricas Globales por Ronda</h2>
          
          <div className="space-y-8">
            {/* Gráfico de Train Loss */}
            <div className="bg-gray-700/50 border border-gray-600 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Pérdida de Entrenamiento (avg_train_loss)</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={metricsData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                  <XAxis 
                    dataKey="round" 
                    label={{ value: 'Ronda', position: 'insideBottomRight', offset: -5 }}
                    stroke="#888"
                  />
                  <YAxis 
                    label={{ value: 'Valor', angle: -90, position: 'insideLeft' }}
                    stroke="#888"
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #4b5563' }}
                    labelStyle={{ color: '#fff' }}
                    formatter={(value) => value.toFixed(6)}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="avg_train_loss" 
                    stroke="#3b82f6" 
                    strokeWidth={2}
                    dot={{ fill: '#3b82f6', r: 4 }}
                    activeDot={{ r: 6 }}
                    name="Train Loss"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Gráfico de Actor Loss */}
            <div className="bg-gray-700/50 border border-gray-600 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Pérdida del Actor (avg_actor_loss)</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={metricsData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                  <XAxis 
                    dataKey="round"
                    label={{ value: 'Ronda', position: 'insideBottomRight', offset: -5 }}
                    stroke="#888"
                  />
                  <YAxis 
                    label={{ value: 'Valor', angle: -90, position: 'insideLeft' }}
                    stroke="#888"
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #4b5563' }}
                    labelStyle={{ color: '#fff' }}
                    formatter={(value) => value.toFixed(6)}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="avg_actor_loss" 
                    stroke="#f59e0b" 
                    strokeWidth={2}
                    dot={{ fill: '#f59e0b', r: 4 }}
                    activeDot={{ r: 6 }}
                    name="Actor Loss"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Gráfico de Validation Reward */}
            <div className="bg-gray-700/50 border border-gray-600 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Recompensa de Validación (avg_val_reward)</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={metricsData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                  <XAxis 
                    dataKey="round"
                    label={{ value: 'Ronda', position: 'insideBottomRight', offset: -5 }}
                    stroke="#888"
                  />
                  <YAxis 
                    label={{ value: 'Valor', angle: -90, position: 'insideLeft' }}
                    stroke="#888"
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #4b5563' }}
                    labelStyle={{ color: '#fff' }}
                    formatter={(value) => value.toFixed(6)}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="avg_val_reward" 
                    stroke="#10b981" 
                    strokeWidth={2}
                    dot={{ fill: '#10b981', r: 4 }}
                    activeDot={{ r: 6 }}
                    name="Validation Reward"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function SessionDetail({ session, onBack }) {
  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString('es-ES', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const calculateDuration = (startTime, endTime) => {
    if (!endTime) return 'En progreso...';
    const start = new Date(startTime);
    const end = new Date(endTime);
    const diffMs = end - start;
    const diffMins = Math.floor(diffMs / 60000);
    const secs = Math.floor((diffMs % 60000) / 1000);
    const mins = diffMins % 60;
    const hours = Math.floor(diffMins / 60);
    
    if (hours > 0) return `${hours}h ${mins}m ${secs}s`;
    if (mins > 0) return `${mins}m ${secs}s`;
    return `${secs}s`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 p-8">
      <div className="max-w-7xl mx-auto">
        <button
          onClick={onBack}
          className="mb-6 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-medium transition"
        >
          ← Volver
        </button>

        <div className="bg-gray-800 border border-gray-700 rounded-2xl p-8 shadow-xl backdrop-blur-sm">
          <div className="flex items-center justify-between mb-8">
            <h1 className="text-3xl font-bold text-white tracking-tight">Sesión #{session.id}</h1>
            <span className={`px-4 py-2 rounded-full text-sm font-bold border ${
              session.status === 'ACTIVE' 
                ? 'bg-blue-500/10 text-blue-400 border-blue-500/30' 
                : 'bg-orange-500/10 text-orange-400 border-orange-500/30'
            }`}>
              {session.status}
            </span>
          </div>

          <div className="space-y-6 mb-8">
            <div>
              <p className="text-sm text-gray-400 mb-2">Hora de inicio</p>
              <p className="text-lg text-white">{formatDate(session.start_time)}</p>
            </div>

            {session.end_time && (
              <div>
                <p className="text-sm text-gray-400 mb-2">Hora de finalización</p>
                <p className="text-lg text-white">{formatDate(session.end_time)}</p>
              </div>
            )}

            <div>
              <p className="text-sm text-gray-400 mb-2">Duración</p>
              <p className="text-lg text-white">{calculateDuration(session.start_time, session.end_time)}</p>
            </div>

            <div>
              <p className="text-sm text-gray-400 mb-2">Total de rondas</p>
              <p className="text-lg text-white font-semibold">{session.total_rounds}</p>
            </div>
          </div>

          <div className="border-t border-gray-700 pt-8">
            <SessionClientsComponent sessionId={session.id} />
          </div>
        </div>
      </div>
    </div>
  );
}