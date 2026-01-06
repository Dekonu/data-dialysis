# Data-Dialysis Dashboard - Deployment Guide

This guide covers deploying the Data-Dialysis Dashboard in production environments.

## Prerequisites

- Docker and Docker Compose installed
- PostgreSQL 16+ (or use provided Docker Compose)
- Minimum 2GB RAM
- Minimum 10GB disk space

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd DataDialysis
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start all services**
   ```bash
   docker-compose up -d
   ```

4. **Verify deployment**
   ```bash
   # Check API health
   curl http://localhost:8000/api/health
   
   # Check frontend
   curl http://localhost:3000
   ```

5. **View logs**
   ```bash
   docker-compose logs -f
   ```

## Environment Variables

### Backend API (.env)

```bash
# Database Configuration
DB_TYPE=postgresql
DB_HOST=postgres
DB_PORT=5432
DB_NAME=datadialysis
DB_USER=datadialysis
DB_PASSWORD=your_secure_password
DB_SSL_MODE=require

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Logging
JSON_LOGS=true
LOG_LEVEL=INFO

# Security
RATE_LIMIT_DEFAULT=100
RATE_LIMIT_WINDOW=60
ENABLE_HSTS=true

# CORS (comma-separated)
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### Frontend (dashboard-frontend/.env.local)

```bash
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
NEXT_PUBLIC_WS_URL=wss://api.yourdomain.com
```

## Production Deployment

### 1. Security Checklist

- [ ] Change all default passwords
- [ ] Enable HTTPS/TLS
- [ ] Set `ENABLE_HSTS=true`
- [ ] Configure CORS origins
- [ ] Set `JSON_LOGS=true` for log aggregation
- [ ] Review rate limits
- [ ] Enable database SSL (`DB_SSL_MODE=require`)

### 2. Database Setup

#### Option A: Use Docker Compose PostgreSQL
```bash
docker-compose up -d postgres
```

#### Option B: External PostgreSQL
1. Create database:
   ```sql
   CREATE DATABASE datadialysis;
   CREATE USER datadialysis WITH PASSWORD 'secure_password';
   GRANT ALL PRIVILEGES ON DATABASE datadialysis TO datadialysis;
   ```

2. Update `.env`:
   ```bash
   DB_HOST=your-postgres-host
   DB_PORT=5432
   DB_SSL_MODE=require
   ```

### 3. Build and Deploy

#### Backend
```bash
# Build image
docker build -t datadialysis-api:latest .

# Run container
docker run -d \
  --name datadialysis-api \
  -p 8000:8000 \
  --env-file .env \
  datadialysis-api:latest
```

#### Frontend
```bash
cd dashboard-frontend

# Build image
docker build -t datadialysis-frontend:latest .

# Run container
docker run -d \
  --name datadialysis-frontend \
  -p 3000:3000 \
  --env-file .env.local \
  datadialysis-frontend:latest
```

### 4. Reverse Proxy (Nginx)

Example Nginx configuration:

```nginx
# API Backend
upstream api_backend {
    server localhost:8000;
}

# Frontend
upstream frontend {
    server localhost:3000;
}

server {
    listen 80;
    server_name api.yourdomain.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    
    location / {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /ws/ {
        proxy_pass http://api_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://frontend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Monitoring

### Health Checks

- **API Health**: `GET /api/health`
- **Frontend**: `GET /`

### Logs

#### View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f frontend
docker-compose logs -f postgres
```

#### Log aggregation (Production)
- Use `JSON_LOGS=true` for structured logging
- Integrate with ELK Stack, Loki, or CloudWatch
- Monitor error rates and response times

### Metrics

The dashboard provides metrics endpoints:
- `/api/metrics/overview` - System overview
- `/api/metrics/security` - Security metrics
- `/api/metrics/performance` - Performance metrics

## Troubleshooting

### API won't start
1. Check database connectivity:
   ```bash
   docker-compose exec api python -c "from src.infrastructure.config_manager import get_database_config; print(get_database_config())"
   ```

2. Check logs:
   ```bash
   docker-compose logs api
   ```

3. Verify environment variables:
   ```bash
   docker-compose exec api env | grep DB_
   ```

### Frontend can't connect to API
1. Verify `NEXT_PUBLIC_API_URL` is correct
2. Check CORS configuration in backend
3. Verify network connectivity:
   ```bash
   docker-compose exec frontend wget -O- http://api:8000/api/health
   ```

### Database connection issues
1. Verify PostgreSQL is running:
   ```bash
   docker-compose ps postgres
   ```

2. Test connection:
   ```bash
   docker-compose exec postgres psql -U datadialysis -d datadialysis -c "SELECT 1"
   ```

3. Check connection pool settings if getting "pool exhausted" errors

## Scaling

### Horizontal Scaling

#### API
```bash
# Scale API instances
docker-compose up -d --scale api=3
```

#### Frontend
```bash
# Scale frontend instances
docker-compose up -d --scale frontend=2
```

### Load Balancing

Use a load balancer (Nginx, HAProxy, or cloud load balancer) in front of multiple API instances.

## Backup and Recovery

### Database Backup
```bash
# Backup
docker-compose exec postgres pg_dump -U datadialysis datadialysis > backup.sql

# Restore
docker-compose exec -T postgres psql -U datadialysis datadialysis < backup.sql
```

### Volume Backup
```bash
# Backup PostgreSQL volume
docker run --rm -v datadialysis_postgres_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/postgres_backup.tar.gz /data
```

## Updates and Rollbacks

### Update Application
```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose build
docker-compose up -d
```

### Rollback
```bash
# Use previous image tag
docker-compose up -d --no-deps api:previous-tag
```

## Security Best Practices

1. **Never commit `.env` files**
2. **Use secrets management** (Docker secrets, Kubernetes secrets, AWS Secrets Manager)
3. **Enable SSL/TLS** for all connections
4. **Regular security updates**: `docker-compose pull && docker-compose up -d`
5. **Monitor logs** for suspicious activity
6. **Limit network exposure**: Use internal networks, expose only necessary ports
7. **Regular backups**: Automated daily backups
8. **Access control**: Use firewall rules, VPN, or private networks

## Performance Tuning

### Database
- Adjust connection pool size based on load
- Add database indexes for frequently queried fields
- Monitor query performance

### API
- Adjust rate limits based on usage patterns
- Monitor response times
- Scale horizontally as needed

### Frontend
- Enable CDN for static assets
- Use Next.js Image optimization
- Monitor bundle size

## Support

For issues or questions:
- Check logs: `docker-compose logs`
- Review health endpoints: `/api/health`
- Check GitHub issues
- Contact support team

