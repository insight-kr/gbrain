import { registerPostgresTests } from '../helpers/test-backends.ts';
await registerPostgresTests(() => import('../google-discovery-migration.test.ts'));
