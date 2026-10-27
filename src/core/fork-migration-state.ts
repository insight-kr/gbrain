/** Per-brain fork receipts are separate from upstream config.version. */
export const FORK_MIGRATION_SEEDS: Readonly<Record<string, string>> = {
  'fork_migration.sinaxis-001-google-discovery-embedding': 'applied',
};

export function isForkMigrationSeed(key: string, value: string): boolean {
  return Object.hasOwn(FORK_MIGRATION_SEEDS, key) && FORK_MIGRATION_SEEDS[key] === value;
}
