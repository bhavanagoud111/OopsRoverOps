# MySQL Database Commands via Docker

Quick reference for viewing and managing data in the RoverOps MySQL database.

## Connect to MySQL

### Interactive MySQL Shell
```bash
docker exec -it roverops-mysql mysql -u roverops_user -proverops_password roverops_db
```

Once connected, you can run SQL commands:
```sql
-- Show all tables
SHOW TABLES;

-- View missions
SELECT * FROM missions;

-- View mission reports
SELECT * FROM mission_reports;

-- View mission logs (latest first)
SELECT * FROM mission_logs ORDER BY timestamp DESC LIMIT 20;

-- Exit
EXIT;
```

## One-Line Commands (Non-Interactive)

### Show All Tables
```bash
docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "SHOW TABLES;"
```

### View All Missions
```bash
docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "SELECT mission_id, goal, status, created_at FROM missions ORDER BY created_at DESC;"
```

### View Mission Reports
```bash
docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "SELECT mission_id, mission_status, completed_steps, total_steps, created_at FROM mission_reports ORDER BY created_at DESC;"
```

### View Recent Activity Logs
```bash
docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "SELECT timestamp, agent_type, level, LEFT(message, 80) as message FROM mission_logs ORDER BY timestamp DESC LIMIT 20;"
```

### View Mission Steps
```bash
docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "SELECT mission_id, step_number, action, target_x, target_y, completed FROM mission_steps ORDER BY mission_id, step_number;"
```

### View Detected Obstacles
```bash
docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "SELECT mission_id, x, y, detected_at FROM mission_obstacles ORDER BY mission_id, detected_at;"
```

## Detailed Queries

### Get Full Mission Details
```bash
docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "
SELECT 
    m.mission_id,
    m.goal,
    m.status,
    m.rover_final_x,
    m.rover_final_y,
    COUNT(DISTINCT ms.id) as total_steps,
    SUM(CASE WHEN ms.completed = 1 THEN 1 ELSE 0 END) as completed_steps,
    COUNT(DISTINCT ml.id) as total_logs,
    m.created_at
FROM missions m
LEFT JOIN mission_steps ms ON m.mission_id = ms.mission_id
LEFT JOIN mission_logs ml ON m.mission_id = ml.mission_id
GROUP BY m.mission_id
ORDER BY m.created_at DESC
LIMIT 10;
"
```

### Get Mission Report with Summary
```bash
docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "
SELECT 
    mr.mission_id,
    m.goal,
    mr.mission_status,
    mr.completed_steps,
    mr.total_steps,
    mr.rover_final_x,
    mr.rover_final_y,
    LEFT(mr.summary, 100) as summary,
    LEFT(mr.outcome, 100) as outcome,
    mr.created_at
FROM mission_reports mr
JOIN missions m ON mr.mission_id = m.mission_id
ORDER BY mr.created_at DESC
LIMIT 5;
"
```

### Count Logs by Agent Type
```bash
docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "
SELECT 
    agent_type,
    level,
    COUNT(*) as count
FROM mission_logs
GROUP BY agent_type, level
ORDER BY agent_type, level;
"
```

### View JSON Data (Mission Photos)
```bash
docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "
SELECT 
    mission_id,
    JSON_LENGTH(mission_photos) as photo_count,
    JSON_EXTRACT(mission_photos, '$[0].url') as first_photo_url
FROM mission_reports
WHERE mission_photos IS NOT NULL
LIMIT 5;
"
```

## Useful MySQL Commands

### Check Database Size
```bash
docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "
SELECT 
    table_name AS 'Table',
    ROUND(((data_length + index_length) / 1024 / 1024), 2) AS 'Size (MB)'
FROM information_schema.TABLES
WHERE table_schema = 'roverops_db'
ORDER BY (data_length + index_length) DESC;
"
```

### Count Records per Table
```bash
docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "
SELECT 
    'missions' as table_name, COUNT(*) as count FROM missions
UNION ALL
SELECT 'mission_steps', COUNT(*) FROM mission_steps
UNION ALL
SELECT 'mission_logs', COUNT(*) FROM mission_logs
UNION ALL
SELECT 'mission_obstacles', COUNT(*) FROM mission_obstacles
UNION ALL
SELECT 'mission_reports', COUNT(*) FROM mission_reports;
"
```

### View Table Structure
```bash
# Missions table
docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "DESCRIBE missions;"

# Mission reports table
docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "DESCRIBE mission_reports;"
```

## Export Data

### Export All Missions to CSV
```bash
docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "
SELECT * FROM missions
" > missions_export.csv
```

### Export Mission Reports
```bash
docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "
SELECT * FROM mission_reports
" > reports_export.csv
```

## Backup & Restore

### Backup Database
```bash
docker exec roverops-mysql mysqldump -u roverops_user -proverops_password roverops_db > roverops_backup.sql
```

### Restore Database
```bash
docker exec -i roverops-mysql mysql -u roverops_user -proverops_password roverops_db < roverops_backup.sql
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `docker exec -it roverops-mysql mysql -u roverops_user -proverops_password roverops_db` | Interactive MySQL shell |
| `docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "SQL"` | Run SQL command |
| `docker compose logs mysql` | View MySQL container logs |
| `docker compose ps mysql` | Check MySQL container status |

## Tips

1. **Remove password warning**: Add `2>/dev/null` to suppress password warnings:
   ```bash
   docker exec roverops-mysql mysql -u roverops_user -proverops_password roverops_db -e "SHOW TABLES;" 2>/dev/null
   ```

2. **Format output**: Use `\G` for vertical output in interactive mode:
   ```sql
   SELECT * FROM missions LIMIT 1\G
   ```

3. **Pretty print JSON**: Use `JSON_PRETTY()` for readable JSON:
   ```sql
   SELECT JSON_PRETTY(mission_photos) FROM mission_reports LIMIT 1;
   ```

