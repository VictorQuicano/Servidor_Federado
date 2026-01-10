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
  const API_URL = process.env.REACT_APP_API_URL;
  
  const fetchSessionClients = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_URL}/training/${sessionId}/client/`);
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
