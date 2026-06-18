-- Run this in the Supabase SQL editor before enabling the authenticated app.
-- It creates teacher profiles and makes saved networks private per user.

create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text not null,
  first_name text not null default '',
  last_name text not null default '',
  teaching_subject text not null default '',
  phone text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.networks
  add column if not exists user_id uuid references auth.users(id) on delete cascade;

create index if not exists networks_user_id_created_at_idx
  on public.networks (user_id, created_at desc);

alter table public.profiles enable row level security;
alter table public.networks enable row level security;

drop policy if exists "Users can read their own profile" on public.profiles;
create policy "Users can read their own profile"
  on public.profiles
  for select
  to authenticated
  using (auth.uid() = id);

drop policy if exists "Users can insert their own profile" on public.profiles;
create policy "Users can insert their own profile"
  on public.profiles
  for insert
  to authenticated
  with check (auth.uid() = id);

drop policy if exists "Users can update their own profile" on public.profiles;
create policy "Users can update their own profile"
  on public.profiles
  for update
  to authenticated
  using (auth.uid() = id)
  with check (auth.uid() = id);

drop policy if exists "Users can read their own networks" on public.networks;
create policy "Users can read their own networks"
  on public.networks
  for select
  to authenticated
  using (auth.uid() = user_id);

drop policy if exists "Users can insert their own networks" on public.networks;
create policy "Users can insert their own networks"
  on public.networks
  for insert
  to authenticated
  with check (auth.uid() = user_id);

drop policy if exists "Users can update their own networks" on public.networks;
create policy "Users can update their own networks"
  on public.networks
  for update
  to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

drop policy if exists "Users can delete their own networks" on public.networks;
create policy "Users can delete their own networks"
  on public.networks
  for delete
  to authenticated
  using (auth.uid() = user_id);
