import React, { useState, useEffect, useRef } from 'react';
import {
  Box, IconButton, Typography, Button, TextField,
  List, ListItem, ListItemButton, ListItemText, Avatar, Paper, Stack,
  CssBaseline, createTheme, ThemeProvider, CircularProgress,
  FormControl, InputLabel, MenuItem, Select
} from '@mui/material';
import { Add as AddIcon,
         Brightness4 as DarkModeIcon, Brightness7 as LightModeIcon, Send as SendIcon,
         SmartToy as BotIcon, Person as UserIcon } from '@mui/icons-material';
import { v4 as uuidv4 } from 'uuid';

// -- (MUI theme same as original) --
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#2196f3' },
    background: { default: '#0a0a0a', paper: '#1a1a1a' },
    text: { primary: '#e0e0e0', secondary: '#bdbdbd' },
  },
  components: {
    MuiButton: { styleOverrides: { root: { borderRadius: '12px' } } },
    MuiPaper: { styleOverrides: { root: { borderRadius: '12px' } } },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': { borderRadius: '12px' },
        },
      },
    },
  },
});

// Define a new light theme
const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#007aff' },
    background: { default: '#f0f2f5', paper: '#ffffff' },
    text: { primary: '#1c1c1e', secondary: '#6c6c70' },
  },
  components: {
    MuiButton: { styleOverrides: { root: { borderRadius: '12px' } } },
    MuiPaper: { styleOverrides: { root: { borderRadius: '12px' } } },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': { borderRadius: '12px' },
        },
      },
    },
  },
});

function App() {
  // We'll use this state to manage all sessions
  const [sessions, setSessions] = useState([]);
  // This state tracks which session is currently active
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [connected, setConnected] = useState(false);
  const [currentInput, setCurrentInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [ragMode, setRagMode] = useState('default');
  const [historyLength, setHistoryLength] = useState(3);
  // State to manage the current theme mode
  const [themeMode, setThemeMode] = useState('dark');
  const chatEndRef = useRef(null); // Ref to the end of the chat messages
  const isMounted = useRef(false);

  const formatDateTime = (value) => {
    if (!value) return '';
    const dateValue = new Date(value);
    return Number.isNaN(dateValue.getTime()) ? value : dateValue.toLocaleString();
  };

  const getModeLabel = (mode) => {
    switch (mode) {
      case 'rewrite query':
        return 'Rewrite query';
      case 'embedding history':
        return 'Embed History';
      case 'ctqe':
        return 'CTQE';
      default:
        return 'Default';
    }
  };

  // Scroll to the bottom of the chat history whenever it updates
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [sessions, activeSessionId]);

  useEffect(() => {
    if (!isMounted.current) {
      handleNewConversation();
      isMounted.current = true;
    }
  }, []);

  // Function to toggle the theme mode
  const toggleTheme = () => {
    setThemeMode(prevMode => (prevMode === 'dark' ? 'light' : 'dark'));
  };
  
  const handleSendMessage = async () => {
    if (!currentInput.trim() || !activeSessionId) return;

    // Get the current session
    const currentSessionIndex = sessions.findIndex(s => s.id === activeSessionId);
    if (currentSessionIndex === -1) return;

    // Add user message to the active session's history
    const sentAt = new Date();
    const requestStartedAt = performance.now();
    const newUserMessage = { id: uuidv4(), author: 'user', content: currentInput };
    const updatedSessionsWithUserMsg = [...sessions];
    updatedSessionsWithUserMsg[currentSessionIndex].chatHistory.push(newUserMessage);
    setSessions(updatedSessionsWithUserMsg);

    const inputToSend = currentInput;
    setCurrentInput('');
    setLoading(true);

    // Add a placeholder assistant message to the active session's history
    const assistantPlaceholderId = uuidv4();
    const updatedSessionsWithPlaceholder = [...updatedSessionsWithUserMsg];
    updatedSessionsWithPlaceholder[currentSessionIndex].chatHistory.push({
      id: assistantPlaceholderId,
      author: 'assistant',
      content: '',
      meta: {
        mode: ragMode,
        sentAt: sentAt.toISOString(),
      }
    });
    setSessions(updatedSessionsWithPlaceholder);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: activeSessionId,
          message: inputToSend,
          rag_mode: ragMode,
          history_length: historyLength,
        })
      });

      const serverTime = response.headers.get('x-server-time');
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantText = "";
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        assistantText += decoder.decode(value, { stream: true });
        
        // Update the content of the assistant message in the active session
        setSessions(prevSessions => prevSessions.map(session => {
          if (session.id === activeSessionId) {
            return {
              ...session,
              chatHistory: session.chatHistory.map(msg =>
                msg.id === assistantPlaceholderId ? {
                  ...msg,
                  content: assistantText,
                  meta: {
                    ...(msg.meta || {}),
                    serverTime: serverTime || msg.meta?.serverTime,
                    receivedAt: msg.meta?.receivedAt || null,
                    latencyMs: msg.meta?.latencyMs || null,
                  }
                } : msg
              )
            };
          }
          return session;
        }));
      }

      const receivedAt = new Date();
      const latencyMs = Math.round(performance.now() - requestStartedAt);
      setSessions(prevSessions => prevSessions.map(session => {
        if (session.id === activeSessionId) {
          return {
            ...session,
            chatHistory: session.chatHistory.map(msg =>
              msg.id === assistantPlaceholderId ? {
                ...msg,
                meta: {
                  ...(msg.meta || {}),
                  serverTime: serverTime || receivedAt.toISOString(),
                  receivedAt: receivedAt.toISOString(),
                  latencyMs,
                }
              } : msg
            )
          };
        }
        return session;
      }));
    } catch (error) {
      console.error("Streaming error:", error);
      // Handle error by setting an error message
      setSessions(prevSessions => prevSessions.map(session => {
        if (session.id === activeSessionId) {
          return {
            ...session,
            chatHistory: session.chatHistory.map(msg =>
              msg.id === assistantPlaceholderId ? { ...msg, content: "Error: Could not get a response." } : msg
            )
          };
        }
        return session;
      }));
    } finally {
      setLoading(false);
    }
  };

  const handleNewConversation = async () => {
    setConnected(false);
    try {
      const res = await fetch("http://localhost:8000/session/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rag_mode: ragMode,
          history_length: historyLength,
        })
      });
      const data = await res.json();
      const newSession = {
        id: data.session_id,
        chatHistory: [{ id: uuidv4(), author: 'assistant', content: data.content }]
      };
      setSessions(prevSessions => [newSession, ...prevSessions]);
      setActiveSessionId(data.session_id);
    } catch (error) {
      console.error("Failed to start a new conversation:", error);
      // You can add a visual indicator for the user here
    } finally {
      setConnected(true);
    }
  };
  
  const handleSelectSession = (sessionId) => {
    setActiveSessionId(sessionId);
  };

  // Get the chat history for the currently active session
  const activeChatHistory = sessions.find(s => s.id === activeSessionId)?.chatHistory || [];

  return (
    <ThemeProvider theme={themeMode === 'dark' ? darkTheme : lightTheme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', height: '100vh', bgcolor: 'background.default', color: 'text.primary' }}>
        {/* Sidebar with New Conversation */}
        <Box sx={{ width: 288, minWidth: 288,bgcolor: 'background.paper', borderRight: '1px solid', borderColor: 'divider', p: 2, display: { xs: 'none', md: 'flex' }, flexDirection: 'column' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', pb: 2, borderBottom: '1px solid', borderColor: 'divider' }}>
            <Typography variant="h6" color="primary" fontWeight="bold">CTQE CHATBOT</Typography>
            <IconButton onClick={handleNewConversation} color="inherit" sx={{ bgcolor: 'primary.main', '&:hover': { bgcolor: 'primary.dark' }, boxShadow: 3 }} title="Create new conversation">
              <AddIcon />
            </IconButton>
          </Box>
          <List sx={{ flexGrow: 1, my: 2, overflowY: 'auto' }}>
            {sessions.map((session) => (
              <ListItem disablePadding key={session.id}>
                <ListItemButton 
                  selected={session.id === activeSessionId}
                  onClick={() => handleSelectSession(session.id)}
                  sx={{ 
                    borderRadius: '8px', 
                    mb: 1, 
                    bgcolor: session.id === activeSessionId ? 'primary.dark' : 'background.paper',
                    '&:hover': { bgcolor: session.id === activeSessionId ? 'primary.dark' : 'rgba(255, 255, 255, 0.08)' }
                  }}
                >
                  <ListItemText
                    primary={`Session: ${session.id.substring(0, 8)}...`}
                    sx={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}
                  />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
          <Box sx={{ mt: 2, p: 1.5, borderRadius: 2, bgcolor: 'action.hover', border: '1px solid', borderColor: 'divider' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="caption" sx={{ letterSpacing: 1, textTransform: 'uppercase', color: 'text.secondary' }}>
                Controls
              </Typography>
              <Button
                onClick={toggleTheme}
                size="small"
                variant="outlined"
                startIcon={themeMode === 'dark' ? <DarkModeIcon /> : <LightModeIcon />}
                sx={{ minWidth: 0, px: 1.2, py: 0.5, borderRadius: 999, textTransform: 'none' }}
              >
                {themeMode === 'dark' ? 'Dark' : 'Light'}
              </Button>
            </Box>

            <Box sx={{ display: 'grid', gap: 1 }}>
              <FormControl fullWidth size="small">
                <InputLabel id="rag-mode-label">Mode</InputLabel>
                <Select
                  labelId="rag-mode-label"
                  value={ragMode}
                  label="Mode"
                  onChange={(e) => setRagMode(e.target.value)}
                >
                  <MenuItem value="rewrite query">Rewrite</MenuItem>
                  <MenuItem value="embedding history">Embed History</MenuItem>
                  <MenuItem value="ctqe">CTQE</MenuItem>
                  <MenuItem value="default">Default</MenuItem>
                </Select>
              </FormControl>

              <TextField
                fullWidth
                size="small"
                label="Turns"
                type="number"
                value={historyLength}
                onChange={(e) => setHistoryLength(Math.max(1, Number(e.target.value) || 1))}
                inputProps={{ min: 1, step: 1 }}
              />
            </Box>
          </Box>
        </Box>

        {/* Main Chat Area */}
        <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', bgcolor: 'background.default'}}>
          {/* Chat history container */}
          <Box sx={{ 
            flexGrow: 1, 
            overflowY: 'auto', 
            p: 3,
            width: '100%',
            boxSizing: 'border-box' 
          }}>
            <Stack spacing={2} sx={{ mb: 2 }}>
              {activeChatHistory.length > 0 ? (
                activeChatHistory.map(msg => (
                  <Box key={msg.id} sx={{ 
                    display: 'flex', 
                    alignItems: 'flex-start',
                    justifyContent: msg.author === 'user' ? 'flex-end' : 'flex-start',
                    gap: 2,
                    width: '100%',
                  }}>
                    {msg.author !== 'user' && (
                      <Avatar sx={{ bgcolor: 'primary.main', width: 40, height: 40 }}><BotIcon /></Avatar>
                    )}
                    <Paper elevation={3} sx={{
                      p: 2,
                      bgcolor: msg.author === 'user' ? 'primary.main' : 'background.paper',
                      color: msg.author === 'user' ? 'white' : 'text.primary',
                      borderRadius: '12px',
                      maxWidth: { xs: '85%', md: '55%' },
                      whiteSpace: 'pre-wrap',
                      overflowWrap: 'break-word',
                      wordWrap: 'break-word',
                    }}>
                      <Typography variant="body1">
                        {msg.content}
                      </Typography>
                      {msg.author === 'assistant' && msg.meta && (
                        <Typography variant="caption" sx={{ display: 'block', mt: 1, color: 'text.secondary' }}>
                          {getModeLabel(msg.meta.mode)} · Sent {formatDateTime(msg.meta.sentAt)} · Received {formatDateTime(msg.meta.receivedAt || msg.meta.serverTime)} · {msg.meta.latencyMs != null ? `${msg.meta.latencyMs} ms` : 'timing pending'}
                        </Typography>
                      )}
                    </Paper>
                    {msg.author === 'user' && (
                      <Avatar sx={{ bgcolor: 'grey.700', width: 40, height: 40 }}><UserIcon /></Avatar>
                    )}
                  </Box>
                ))
              ) : (
                <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Typography variant="body1" color="text.secondary">
                    {!connected ? 'Connecting to server...' : 'Send a message to start the conversation.'}
                  </Typography>
                </Box>
              )}
              {loading && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, justifyContent: 'flex-start', mt: 2 }}>
                  <Avatar sx={{ bgcolor: 'primary.main' }}><BotIcon /></Avatar>
                  <CircularProgress size={24} />
                </Box>
              )}
              <div ref={chatEndRef} />
            </Stack>
          </Box>

          {/* Input area */}
          <Box sx={{ 
            flexShrink: 0, 
            p: 2, 
            bgcolor: 'background.paper', 
            borderTop: '1px solid', 
            borderColor: 'divider', 
            display: 'flex', 
            gap: 2,
            width: '100%',
            boxSizing: 'border-box'
          }}>
            <TextField
              fullWidth
              multiline 
              maxRows={4}
              variant="outlined"
              placeholder="Ask me something..."
              value={currentInput}
              onChange={(e) => setCurrentInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              disabled={!connected || loading}
              sx={{
                '& .MuiOutlinedInput-notchedOutline': { borderColor: 'divider' },
                '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: 'primary.main' },
                '&.Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: 'primary.main' },
                bgcolor: 'background.default',
                color: 'text.primary'
              }}
            />
            <IconButton
              onClick={handleSendMessage}
              color="primary"
              disabled={!connected || loading || !currentInput.trim()}
              title="Send message"
              sx={{ bgcolor: 'primary.main', color: 'white', '&:hover': { bgcolor: 'primary.dark' }, '&:disabled': { bgcolor: 'grey.800' }, boxShadow: 3, p: '15px' }}
            >
              <SendIcon />
            </IconButton>
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
