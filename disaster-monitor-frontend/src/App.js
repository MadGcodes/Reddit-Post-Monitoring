import React, { useState, useEffect } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import { StyledEngineProvider } from '@mui/material/styles';
import { createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { 
  Container, Paper, Typography, CircularProgress, Grid, Card, CardContent, 
  CardMedia, Box, Button, Snackbar, Alert, AppBar, Toolbar, Divider,
  Chip, FormControl, InputLabel, Select, MenuItem, Tabs, Tab, IconButton,
  List, ListItem, ListItemText, ListItemSecondaryAction, Avatar, Link
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Timeline as TimelineIcon,
  CloudQueue as CloudIcon,
  Article as ArticleIcon,
  Search as SearchIcon,
  Mood as MoodIcon,
  Forum as ForumIcon
} from '@mui/icons-material';
import './App.css';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#2196f3' },
    secondary: { main: '#03a9f4' },
    background: { default: '#f5f7fa', paper: '#ffffff' },
    error: { main: '#ef5350' },
    warning: { main: '#ff9800' },
    info: { main: '#03a9f4' },
    success: { main: '#4caf50' },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: { fontWeight: 600 },
    h5: { fontWeight: 500 },
    h6: { fontWeight: 500 },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          transition: 'all 0.3s ease-in-out',
          '&:hover': {
            transform: 'translateY(-4px)',
            boxShadow: '0 8px 24px rgba(0,0,0,0.1)',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          padding: '8px 16px',
        },
        contained: {
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 6,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
        },
      },
    },
  },
});

const defaultKeywords = [
  'earthquake', 'tsunami', 'flood', 'hurricane', 'tornado',
  'wildfire', 'volcanic', 'landslide', 'storm surge', 'cyclone',
  'typhoon', 'avalanche', 'mudslide', 'drought', 'forest fire',
  'bushfire', 'seismic', 'tremor', 'evacuation'
];

function TabPanel({ children, value, index }) {
  return (
    <div role="tabpanel" hidden={value !== index}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

function App() {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [selectedKeywords, setSelectedKeywords] = useState([]);
  const [timeRange, setTimeRange] = useState(24);
  const [tabValue, setTabValue] = useState(0);

  useEffect(() => {
    return () => {
      setData(null);
      setError(null);
    };
  }, []);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const fetchData = async () => {
    if (selectedKeywords.length === 0) {
      setError('Please select at least one keyword');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:5000/api/fetch-and-analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          keywords: selectedKeywords, 
          hours_back: timeRange 
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      if (result.success) {
        setData(result);
      } else {
        setError(result.error || 'An error occurred while fetching data');
      }
    } catch (error) {
      setError(error.message || 'Failed to connect to the server. Please check if the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const PostCard = ({ post }) => (
    <Card sx={{ mb: 2, '&:hover': { transform: 'none' } }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
          <Avatar sx={{ bgcolor: theme.palette.primary.main }}>
            {post.subreddit[0].toUpperCase()}
          </Avatar>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h6" gutterBottom>
              {post.title}
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              {post.text.length > 200 ? `${post.text.substring(0, 200)}...` : post.text}
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                r/{post.subreddit} • Score: {post.score} • Comments: {post.num_comments}
              </Typography>
              <Button 
                variant="outlined" 
                size="small" 
                href={post.url}
                target="_blank"
                rel="noopener noreferrer"
                startIcon={<ArticleIcon />}
              >
                View Post
              </Button>
            </Box>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );

  const VisualizationCard = ({ title, imageData, description, icon: Icon }) => {
    if (!imageData) {
      return (
        <Card sx={{ height: '100%', boxShadow: 3 }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
              <Icon color="primary" />
              <Typography variant="h6" color="primary">{title}</Typography>
            </Box>
            <Typography variant="body2" color="error">Visualization data not available</Typography>
          </CardContent>
        </Card>
      );
    }

    return (
      <Card sx={{ height: '100%', boxShadow: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <Icon color="primary" />
            <Typography variant="h6" color="primary">{title}</Typography>
          </Box>
          {description && (
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              {description}
            </Typography>
          )}
        </CardContent>
        <CardMedia 
          component="img" 
          sx={{ objectFit: 'contain', maxHeight: 400, p: 2 }} 
          image={`data:image/png;base64,${imageData}`} 
          alt={title}
          onError={(e) => {
            e.target.onerror = null;
            e.target.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
          }}
        />
      </Card>
    );
  };

  return (
    <StyledEngineProvider injectFirst>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <AppBar position="static" elevation={0}>
          <Toolbar>
            <Typography variant="h5" sx={{ flexGrow: 1, fontWeight: 'bold' }}>
              Natural Disaster Monitor
            </Typography>
            {data && (
              <IconButton color="inherit" onClick={fetchData} disabled={loading}>
                <RefreshIcon />
              </IconButton>
            )}
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
          {!data ? (
            <Box sx={{ textAlign: 'center', mt: 8 }}>
              <Paper 
                sx={{ 
                  p: 4, 
                  maxWidth: 800, 
                  mx: 'auto',
                  background: 'linear-gradient(to bottom right, #ffffff, #f8f9fa)'
                }}
              >
                <Typography variant="h4" color="primary" sx={{ mb: 3 }}>
                  Reddit Disaster Monitor
                </Typography>
                <Typography variant="body1" sx={{ mb: 4 }}>
                  Select keywords and time range to monitor disaster-related content on Reddit.
                </Typography>
                
                <Grid container spacing={3} sx={{ mb: 4 }}>
                  <Grid item xs={12}>
                    <FormControl fullWidth>
                      <InputLabel>Select Keywords</InputLabel>
                      <Select
                        multiple
                        value={selectedKeywords}
                        onChange={(e) => setSelectedKeywords(e.target.value)}
                        renderValue={(selected) => (
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                            {selected.map((value) => (
                              <Chip key={value} label={value} />
                            ))}
                          </Box>
                        )}
                      >
                        {defaultKeywords.map((keyword) => (
                          <MenuItem key={keyword} value={keyword}>
                            {keyword}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12}>
                    <FormControl fullWidth>
                      <InputLabel>Time Range</InputLabel>
                      <Select
                        value={timeRange}
                        label="Time Range"
                        onChange={(e) => setTimeRange(e.target.value)}
                      >
                        <MenuItem value={6}>Last 6 hours</MenuItem>
                        <MenuItem value={12}>Last 12 hours</MenuItem>
                        <MenuItem value={24}>Last 24 hours</MenuItem>
                        <MenuItem value={48}>Last 48 hours</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                </Grid>

                <Button 
                  variant="contained" 
                  color="primary" 
                  size="large" 
                  onClick={fetchData} 
                  disabled={loading || selectedKeywords.length === 0}
                  startIcon={<SearchIcon />}
                  sx={{ 
                    mt: 2,
                    px: 4,
                    py: 1.5,
                    fontSize: '1.1rem'
                  }}
                >
                  {loading ? <CircularProgress size={24} color="inherit" /> : 'Analyze Posts'}
                </Button>
              </Paper>
            </Box>
          ) : (
            <>
              <Paper sx={{ mb: 3 }}>
                <AppBar position="static" color="default" elevation={0}>
                  <Tabs
                    value={tabValue}
                    onChange={handleTabChange}
                    indicatorColor="primary"
                    textColor="primary"
                    variant="fullWidth"
                  >
                    <Tab icon={<ArticleIcon />} label="Posts" />
                    <Tab icon={<TimelineIcon />} label="Visualizations" />
                  </Tabs>
                </AppBar>

                <TabPanel value={tabValue} index={0}>
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Found {data.posts.length} posts
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {selectedKeywords.map((keyword) => (
                        <Chip 
                          key={keyword}
                          label={keyword}
                          color="primary"
                          variant="outlined"
                        />
                      ))}
                    </Box>
                  </Box>
                  <Box>
                    {data.posts.map((post, index) => (
                      <PostCard key={index} post={post} />
                    ))}
                  </Box>
                </TabPanel>

                <TabPanel value={tabValue} index={1}>
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <VisualizationCard 
                        title="Word Cloud"
                        imageData={data.visualizations.word_cloud}
                        description="Most frequent terms in disaster-related posts"
                        icon={CloudIcon}
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <VisualizationCard 
                        title="Post Frequency"
                        imageData={data.visualizations.time_series}
                        description="Number of posts over time"
                        icon={TimelineIcon}
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <VisualizationCard 
                        title="Sentiment Analysis"
                        imageData={data.visualizations.sentiment_analysis}
                        description="Distribution of post sentiments (Positive/Negative/Neutral)"
                        icon={MoodIcon}
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <VisualizationCard 
                        title="Subreddit Distribution"
                        imageData={data.visualizations.subreddit_distribution}
                        description="Distribution of posts across different subreddits"
                        icon={ForumIcon}
                      />
                    </Grid>
                  </Grid>
                </TabPanel>
              </Paper>

              <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
                <Button 
                  variant="outlined" 
                  color="primary" 
                  onClick={() => setData(null)}
                  startIcon={<SearchIcon />}
                >
                  New Analysis
                </Button>
                <Button 
                  variant="contained" 
                  color="primary" 
                  onClick={fetchData}
                  disabled={loading}
                  startIcon={<RefreshIcon />}
                >
                  Refresh Data
                </Button>
              </Box>
            </>
          )}
        </Container>

        <Snackbar 
          open={!!error} 
          autoHideDuration={6000} 
          onClose={() => setError(null)}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <Alert 
            onClose={() => setError(null)} 
            severity="error" 
            variant="filled"
            sx={{ width: '100%' }}
          >
            {error}
          </Alert>
        </Snackbar>
      </ThemeProvider>
    </StyledEngineProvider>
  );
}

export default App;
