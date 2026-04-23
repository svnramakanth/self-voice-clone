import Link from "next/link";

export function Nav() {
  return (
    <nav className="nav">
      <Link href="/" className="nav-brand">VClone</Link>
      <div className="nav-links">
        <Link href="/">Start Here</Link>
        <Link href="/enrollment">1. Save Voice</Link>
        <Link href="/voices">2. Saved Profiles</Link>
        <Link href="/synthesis">3. Generate</Link>
      </div>
    </nav>
  );
}
