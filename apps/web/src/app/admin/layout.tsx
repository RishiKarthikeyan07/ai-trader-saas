import { redirect } from 'next/navigation';

// Admin layout - add auth check here
export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // TODO: Add admin auth check
  // const { user } = await getUser();
  // if (!user?.is_admin) redirect('/');

  return (
    <div className="min-h-screen bg-graphite-950">
      <div className="border-b border-border/50 bg-background/95 backdrop-blur">
        <div className="container mx-auto px-6 py-4">
          <h1 className="text-2xl font-bold text-accent-red">
            🔐 Admin Panel
          </h1>
        </div>
      </div>
      <div className="container mx-auto p-6">{children}</div>
    </div>
  );
}
